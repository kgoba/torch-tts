import torch
import torch.nn as nn
from decoder_cell import DecoderCell, Taco1DecoderCell, Taco2DecoderCell
from decoder import Decoder
from encoder import Encoder, Encoder2
from modules.modules import MelPostnet, MelPostnet2
from modules.autoencoder import ReferenceEncoderVAE
from modules.activations import isrlu, isru


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def lengths_to_mask(lengths):
    mask = [torch.ones(x, dtype=torch.bool, device=lengths.device) for x in lengths]
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
    return mask


class Tacotron(nn.Module):
    def __init__(self, encoder, decoder, postnet=None, refencoder=None):
        super().__init__()
        self.refencoder = refencoder
        if refencoder != None:
            self.fc_mem = nn.Linear(refencoder.dim_out, encoder.dim_out, bias=True)
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
        assert cond.dtype == torch.long
        assert cond_lengths.dtype == torch.long
        # assert x == None or x.dtype == torch.float32

        memory = self.encoder(cond, cond_lengths)
        if xref != None and self.refencoder != None:
            style_encoding, kl_loss = self.refencoder(xref, xref_lengths)
            memory = memory + isru(self.fc_mem(style_encoding).unsqueeze(1))
            kl_loss = kl_loss.mean()
        else:
            kl_loss = torch.scalar_tensor(0)

        mmask = None  # lengths_to_mask(cond_lengths)
        y, s, w = self.decoder(memory, mmask, x, max_steps)

        # y_post = self.postnet(y) if self.postnet else y
        y_post = y
        return y, y_post, s, {"w": w, "kl_loss": kl_loss}


def loss_fn(y, x, mask=None, order=1):
    if order == 0:
        loss = y - x
        loss = torch.where(loss > 0, loss, -loss)
        loss = loss * x
    elif order == 1:
        loss = torch.nn.functional.l1_loss(x, y, reduction="none")
    else:
        loss = torch.nn.functional.mse_loss(x, y, reduction="none")

    if mask is None:
        loss = torch.mean(loss)
    else:
        loss = torch.mean(loss * mask, dim=2)
        loss = loss.sum() / mask.sum()

    return loss if order == 1 else loss.sqrt()


def alignment_loss(w):
    D = w.shape[2]
    t = torch.arange(D).unsqueeze(0).unsqueeze(0).to(device=w.device)
    w_var = torch.sum(w * t.square(), axis=2) - torch.sum(w * t, axis=2).square()
    w_std = torch.maximum(w_var, 1e-6 * torch.ones_like(w_var)).sqrt()
    return w_std


def run_training_step(model, batch, device):
    c, c_lengths, x, x_lengths = [t.to(device) for t in batch if isinstance(t, torch.Tensor)]

    xmask = lengths_to_mask(x_lengths).unsqueeze(2)

    y, y_post, s, out_dict = model(c, c_lengths, x, x_lengths, xref=x, xref_lengths=x_lengths)
    T = y.shape[1]
    x, xmask = x[:, :T, :], xmask[:, :T]

    loss_mel = loss_fn(y, x, xmask, order=0)
    loss_mel_post = loss_fn(y_post, x, xmask, order=0)

    pos_weight = torch.Tensor([0.1]).to(device=s.device)
    loss_stop = torch.nn.functional.binary_cross_entropy_with_logits(
        s, xmask.float(), pos_weight=pos_weight  # stop_weight,
    )

    loss = 0.8 * loss_mel + 0.2 * loss_mel_post + 0.1 * loss_stop
    # loss += 0.0002 * out_dict["kl_loss"]
    # loss += 0.0002 * alignment_loss(out_dict["w"]).mean()

    return loss, {
        "loss_mel_db": 120 * (loss_mel.item()),
        "loss_mel_post_db": 120 * (loss_mel_post.item()),
        "loss_stop": loss_stop.item(),
        "loss_kl": out_dict["kl_loss"].item(),
        "w": out_dict["w"].detach().cpu(),
    }


def run_inference_step(model, text_encoder, batch, device, xref=None, max_steps=400):
    with torch.no_grad():
        encoded_text = [text_encoder.encode(text) for text in batch]
        input = [torch.LongTensor(text) for text in encoded_text]
        input_lengths = torch.LongTensor([len(text) for text in encoded_text])
        input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)
        xref_lengths = torch.LongTensor([len(x) for x in xref]) if xref != None else None

        # input = input.to(device, non_blocking=True)
        y, y_post, s, out_dict = model(
            input, input_lengths, xref=xref, xref_lengths=xref_lengths, max_steps=max_steps
        )

        # model_traced = torch.jit.trace(model, (input, imask)) # Export to TorchScript
        return y_post.detach().cpu(), {"s": s.detach().cpu(), "w": out_dict["w"].detach().cpu()}


def build_tacotron(config):
    text_config = config["text"]
    audio_config = config["audio"]
    decoder_config = config["model"]["decoder"]
    encoder_config = config["model"]["encoder"]
    postnet_config = config["model"]["postnet"]

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

    encoder = Encoder2(
        1 + len(text_config["alphabet"]),
        dim_out=encoder_config["dim_out"],
        dim_emb=encoder_config["dim_emb"],
    )

    if postnet_config:
        # postnet = MelPostnet2(
        #     audio_config["num_mels"],
        #     num_layers=2,
        # )
        postnet = MelPostnet(
            # encoder_config["dim_out"],
            audio_config["num_mels"],
            audio_config["num_mels"],
            dim_hidden=postnet_config["dim_hidden"],
            num_layers=postnet_config["num_layers"],
        )
    else:
        postnet = None

    # refencoder = ReferenceEncoderVAE(audio_config["num_mels"], dim_conv=[256, 256], dim_out=16)
    refencoder = None

    return Tacotron(encoder, decoder, postnet=postnet, refencoder=refencoder)
