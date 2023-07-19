import torch
import torch.nn as nn
from modules import PreNet, ResGRUCell, ContentGeneralAttention
from decoder import Decoder, DecoderCell
from encoder import Encoder


class Tacotron(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, cond, cmask, x = None):
        encoder_outputs = self.encoder(cond)
        return self.decoder(encoder_outputs, x)


def build_tacotron(config):
    text_config = config["text"]
    audio_config = config["audio"]
    decoder_config = config["model"]["decoder"]
    encoder_config = config["model"]["encoder"]
    decoder_cell = DecoderCell(encoder_config["dim_enc"], audio_config["num_mels"], decoder_config["r"])
    decoder = Decoder(decoder_cell)
    # decoder = Decoder(encoder_config["dim_enc"], audio_config["num_mels"], decoder_config["r"])
    encoder = Encoder(len(text_config["alphabet"]))
    return Tacotron(encoder, decoder)
