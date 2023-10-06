import torch
import torch.nn as nn
from decoder_cell import DecoderCell, Taco1DecoderCell, Taco2DecoderCell
from decoder import Decoder
from encoder import Encoder, Encoder2
from modules.modules import MelPostnet, MelPostnet2
from modules.style import GST, GST_VAE, VAE
from modules.activations import isrlu, isru
from data.util import lengths_to_mask


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.xavier_normal_(m.weight, gain=1.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Tacotron(nn.Module):
    def __init__(self, encoder, decoder, postnet=None, refencoder=None):
        super().__init__()
        self.refencoder = refencoder
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
        self.apply(weights_init)

    def forward(
        self,
        cond,
        cond_lengths,
        x=None,
        x_lengths=None,
        xref=None,
        xref_lengths=None,
        max_steps: int = 0,
    ):
        memory = self.encoder(cond, cond_lengths)

        kl_loss = torch.scalar_tensor(0)
        if xref != None and self.refencoder != None:
            style_embed, style_loss_dict = self.refencoder(xref, xref_lengths)
            memory += style_embed
            if "kl" in style_loss_dict:
                kl_loss = style_loss_dict["kl"].mean()

        mmask = lengths_to_mask(cond_lengths)
        y, s, w = self.decoder(memory, mmask, x, max_steps, p_no_forcing=0.1)
        # y = y.detach()
        # s = s.detach()
        # w = w.detach()

        y_post = self.postnet(y) if self.postnet else y
        # y_post = y
        return y, y_post, s, {"w": w, "kl_loss": kl_loss}


def mel_loss_fn(y, x, mask=None, order=1):
    if order == 0:
        vol = x.detach().mean(dim=2).unsqueeze(2)
        vol_weight = vol.clip(min=0.1)
        loss = y - x
        loss = torch.where(loss > 0, vol_weight * loss, -loss)
        # loss = loss * x.mean(dim=2).unsqueeze(2)
        # loss = loss / x.mean()
        # loss += 0.2 * (y - x).mean().abs()
    elif order == 1:
        loss = torch.nn.functional.l1_loss(x, y, reduction="none")
    else:
        loss = torch.nn.functional.mse_loss(x, y, reduction="none")

    if mask is None:
        loss = torch.mean(loss)
    else:
        loss = torch.mean(loss * mask, dim=2)
        loss = loss.sum() / mask.sum()

    # if order == 0:
    #     return x.mean() * loss
    if order == 1 or order == 0:
        return loss
    else:
        return loss.sqrt()  # if order == 2 else loss


def alignment_max_loss(w):
    w_max, _ = w.max(axis=2)
    return (1 - w_max).mean()


def alignment_std_loss(w):
    D = w.shape[2]
    t = torch.arange(D).unsqueeze(0).unsqueeze(0).to(device=w.device)
    w_var = torch.sum(w * t.square(), axis=2) - torch.sum(w * t, axis=2).square()
    w_std = torch.clip(w_var, min=0).mean().sqrt()
    return w_std


def run_training_step(model, batch, device):
    c, c_lengths, x, x_lengths = [t.to(device) for t in batch if isinstance(t, torch.Tensor)]

    xmask = lengths_to_mask(x_lengths).unsqueeze(2)

    y, y_post, s, out_dict = model(c, c_lengths, x, x_lengths, xref=x, xref_lengths=x_lengths)
    T = y.shape[1]
    x, xmask = x[:, :T, :], xmask[:, :T]

    loss_mel = mel_loss_fn(y, x, xmask, order=1) + mel_loss_fn(
        y.diff(dim=1), x.diff(dim=1), order=1
    )
    loss_mel_post = mel_loss_fn(y_post, x, xmask, order=1) + mel_loss_fn(
        y_post.diff(dim=1), x.diff(dim=1), order=1
    )

    pos_weight = torch.Tensor([0.1]).to(device=s.device)
    loss_stop = torch.nn.functional.binary_cross_entropy_with_logits(
        s, xmask.float(), pos_weight=pos_weight  # stop_weight,
    )
    loss_w = alignment_std_loss(out_dict["w"])
    loss_kl = out_dict["kl_loss"]

    loss = 0.8 * loss_mel + 0.2 * loss_mel_post + 0.1 * loss_stop
    loss += 0.0002 * loss_kl
    loss += 0.0001 * loss_w

    return loss, {
        "loss_mel_db": 100 * (loss_mel.item()),
        "loss_mel_post_db": 100 * (loss_mel_post.item()),
        "loss_stop": loss_stop.item(),
        "loss_kl": loss_kl.item(),
        "loss_w": loss_w.item(),
        "w": out_dict["w"].detach().cpu(),
    }


def run_inference_step(model, text_encoder, text_batch, device, xref=None, max_steps=400):
    with torch.no_grad():
        encoded_text = [text_encoder.encode(text) for text in text_batch]
        c_lengths = torch.LongTensor([len(text) for text in encoded_text])
        c = [torch.LongTensor(text) for text in encoded_text]
        c = torch.nn.utils.rnn.pad_sequence(c, batch_first=True)
        xref_lengths = torch.LongTensor([len(x) for x in xref]) if xref != None else None

        # input = input.to(device, non_blocking=True)
        y, y_post, s, out_dict = model(
            c, c_lengths, xref=xref, xref_lengths=xref_lengths, max_steps=max_steps
        )

        # model_traced = torch.jit.trace(model, (input, imask)) # Export to TorchScript
        return y_post.detach().cpu(), {"s": s.detach().cpu(), "w": out_dict["w"].detach().cpu()}


def build_tacotron(config):
    text_config = config["text"]
    audio_config = config["audio"]
    decoder_config = config["model"]["decoder"]
    encoder_config = config["model"]["encoder"]

    if decoder_config["type"] == "tacotron1":
        decoder_cell_class = Taco1DecoderCell
    elif decoder_config["type"] == "tacotron2":
        decoder_cell_class = Taco2DecoderCell
    else:
        decoder_cell_class = DecoderCell

    decoder_cell = decoder_cell_class(
        encoder_config["dim_out"],
        audio_config["num_mels"],
        r=decoder_config["r"],
        dim_rnn=decoder_config["dim_rnn"],
        dim_pre=decoder_config["dim_pre"],
        dim_att=decoder_config["dim_att"],
    )

    decoder = Decoder(decoder_cell, decoder_config["r"], audio_config["num_mels"])

    alphabet_size = 1 + len(text_config["alphabet"])
    if "phonemes" in text_config:
        alphabet_size += len(text_config["phonemes"])

    encoder = Encoder2(
        alphabet_size,
        dim_out=encoder_config["dim_out"],
        dim_emb=encoder_config["dim_emb"],
    )

    postnet_config = config["model"].get("postnet")
    if postnet_config:
        if postnet_config.get("type") == "tacotron2":
            postnet = MelPostnet(
                audio_config["num_mels"],
                dim_hidden=postnet_config["dim_hidden"],
                num_layers=postnet_config["num_layers"],
            )
        else:
            postnet = MelPostnet2(
                audio_config["num_mels"],
                dim_hidden=postnet_config["dim_hidden"],
                num_layers=postnet_config["num_layers"],
            )
    else:
        postnet = None

    style_encoder_config = config["model"].get("style_encoder")
    if style_encoder_config:
        refencoder = VAE(
            num_mels=audio_config["num_mels"],
            dim_vae=style_encoder_config["dim_vae"]
        )
    else:
        refencoder = None

    return Tacotron(encoder, decoder, postnet=postnet, refencoder=refencoder)
