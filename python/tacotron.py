import torch
import torch.nn as nn
from decoder import Decoder, DecoderCell
from encoder import Encoder, Encoder2


class Tacotron(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, cond, cmask, x=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
        assert cond.dtype == torch.long
        assert cmask.dtype == torch.bool
        # assert x == None or x.dtype == torch.float32
        memory = self.encoder(cond, cmask)
        return self.decoder(memory, cmask, x)


def build_tacotron(config):
    text_config = config["text"]
    audio_config = config["audio"]
    decoder_config = config["model"]["decoder"]
    encoder_config = config["model"]["encoder"]

    decoder_cell = DecoderCell(
        encoder_config["dim_out"],
        audio_config["num_mels"],
        r=decoder_config["r"],
        dim_pre=decoder_config["dim_pre"],
        dim_att=decoder_config["dim_att"]
    )
    decoder = Decoder(decoder_cell)

    encoder = Encoder2(
        1 + len(text_config["alphabet"]),
        dim_out=encoder_config["dim_out"],
        dim_emb=encoder_config["dim_emb"],
    )
    return Tacotron(encoder, decoder)
