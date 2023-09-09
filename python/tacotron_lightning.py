from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from decoder_cell import DecoderCell, Taco1DecoderCell, Taco2DecoderCell
from decoder import Decoder
from encoder import Encoder, Encoder2
from modules.modules import MelPostnet, MelPostnet2
from modules.autoencoder import ReferenceEncoderVAE
from modules.activations import isrlu, isru
import pytorch_lightning as pl


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


def mel_loss_fn(y, x, mask=None, order=1):
    if order == 1:
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
    # t = torch.linspace(0, 1, D).unsqueeze(0).unsqueeze(0).to(device=w.device)
    t = torch.arange(D).unsqueeze(0).unsqueeze(0).to(device=w.device)
    w_std = torch.sum(w * t.square(), axis=2) - torch.sum(w * t, axis=2).square()
    loss = torch.maximum(w_std, 1e-2 * torch.ones_like(w_std))  # .sqrt()
    return loss


class TacotronTask(pl.LightningModule):
    def __init__(self, encoder, decoder, postnet=None, refencoder=None):
        super().__init__()
        self.refencoder = refencoder
        if refencoder != None:
            self.fc_mem = nn.Linear(refencoder.dim_out, encoder.dim_out, bias=True)
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
        self.apply(weights_init)
        self.max_steps = 400

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        return optimizer

    def predict_forward(self, text_encoder, text_batch, xref=None, max_steps=400):
        encoded_text = [text_encoder.encode(text) for text in text_batch]
        c_lengths = torch.LongTensor([len(text) for text in encoded_text])
        c = [torch.LongTensor(text) for text in encoded_text]
        c = torch.nn.utils.rnn.pad_sequence(c, batch_first=True)
        # xref_lengths = torch.LongTensor([len(x) for x in xref]) if xref != None else None

        mmask = None  # lengths_to_mask(c_lengths)
        memory = self.encoder(c, c_lengths)
        y, s, w = self.decoder(memory, mmask, max_steps=max_steps)
        y_post = self.postnet(y) if self.postnet else y

        return y_post.detach().cpu()

    def train_forward(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        c, c_lengths, x, x_lengths = batch

        x_lengths = torch.clamp(x_lengths, max=500)
        xmask = lengths_to_mask(x_lengths).unsqueeze(2)

        T = xmask.shape[1]
        x = x[:, :T, :]

        mmask = None  # lengths_to_mask(c_lengths)
        memory = self.encoder(c, c_lengths)
        y, s, w = self.decoder(memory, mmask, x)
        y_post = self.postnet(y) if self.postnet else y

        T = y.shape[1]
        x, xmask = x[:, :T, :], xmask[:, :T]
        loss_mel = mel_loss_fn(y, x, xmask, order=2)
        loss_mel_post = mel_loss_fn(y_post, x, xmask, order=2)

        pos_weight = torch.Tensor([0.1]).to(device=s.device)
        loss_stop = torch.nn.functional.binary_cross_entropy_with_logits(
            s, xmask.float(), pos_weight=pos_weight  # stop_weight,
        )

        loss = 0.8 * loss_mel + 0.2 * loss_mel_post + 0.1 * loss_stop
        # loss += 0.0002 * alignment_loss(w).mean()

        return loss

    def training_step(self, batch):
        loss = self.train_forward(batch)
        # Logging to TensorBoard (if installed) by default
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss = self.train_forward(batch)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        return loss


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

    return TacotronTask(encoder, decoder, postnet=postnet, refencoder=refencoder)
