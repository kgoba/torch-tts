import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device= "cpu"

print(device)

try:
    from torchaudio.functional import forced_align
except ImportError:
    print(
        "Failed to import the forced alignment API. "
        "Please install torchaudio nightly builds. "
        "Please refer to https://pytorch.org/get-started/locally "
        "for instructions to install a nightly build.")
    raise

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)

SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

def get_model_en(device):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    return model

def get_emissions_en(model, waveform, device):
    labels = bundle.get_labels()
    with torch.inference_mode():
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)
    return emissions

def get_model_uni(device):
from torchaudio.models import wav2vec2_model

model = wav2vec2_model(
    extractor_mode="layer_norm",
    extractor_conv_layer_config=[
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 2),
        (512, 2, 2),
    ],
    extractor_conv_bias=True,
    encoder_embed_dim=1024,
    encoder_projection_dropout=0.0,
    encoder_pos_conv_kernel=128,
    encoder_pos_conv_groups=16,
    encoder_num_layers=24,
    encoder_num_heads=16,
    encoder_attention_dropout=0.0,
    encoder_ff_interm_features=4096,
    encoder_ff_interm_dropout=0.1,
    encoder_dropout=0.0,
    encoder_layer_norm_first=True,
    encoder_layer_drop=0.1,
    aux_num_out=31,
)

torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt", "model.pt")
checkpoint = torch.load("model.pt", map_location="cpu")

model.load_state_dict(checkpoint)
model.eval()


waveform, _ = torchaudio.load(SPEECH_FILE)
emission = emissions[0].cpu().detach()

# print(labels)
plt.imshow(emission.T)
plt.colorbar()
plt.title("Frame-wise class probability")
plt.xlabel("Time")
plt.ylabel("Labels")
plt.show()
