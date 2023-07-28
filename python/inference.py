import torch
import numpy as np
import torchaudio
import yaml, os, sys, logging

from train_util import find_available_device, Trainer
from dataset import TextEncoder, m_rev
from tacotron import build_tacotron
from audio import AudioFrontend, AudioFrontendConfig

logger = logging.getLogger(__name__)


def run_inference_step(model, text_encoder, batch, device):
    encoded_text = [text_encoder.encode(text) for text in batch]
    input = [torch.LongTensor(text) for text in encoded_text]
    imask = [torch.ones(len(text), dtype=torch.bool) for text in encoded_text]
    imask = torch.nn.utils.rnn.pad_sequence(imask, batch_first=True)
    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)

    y, s, w = model(input, imask)

    return y.detach(), {"s": s.detach(), "w": w.detach()}


def synth_audio(y, audio_frontend):
    # wave = audio_frontend.mel_inv(m_rev(y))
    wave = []
    for y_i in y:
        M_db = m_rev(y_i)
        D_db = audio_frontend.mel_inv(M_db)
        print(M_db.mean(), D_db.mean())
        wave_i = audio_frontend.decode(D_db)
        wave_i = wave_i / wave_i.abs().max()
        wave.append(wave_i)
    return torch.stack(wave)


def main(args):
    config_path = args.config

    config = yaml.safe_load(open(config_path))
    random_seed = config["seed"] or 42

    device = find_available_device()
    print(f"Using device {device}")
    print(f"Number of CPUs: {os.cpu_count()}")

    device = "cpu"  # set PYTORCH_ENABLE_MPS_FALLBACK=1

    text_encoder = TextEncoder(
        config["text"]["alphabet"],
        config["text"].get("character_map"),
        bos=config["text"].get("bos_symbols"),
        eos=config["text"].get("eos_symbols"),
    )

    audio_config = AudioFrontendConfig()
    audio_config.from_json(config["audio"])
    audio_frontend = AudioFrontend(audio_config)

    model = build_tacotron(config)
    model.eval()

    trainer = Trainer(model, args.checkpoint_dir)
    trainer.load_checkpoint(trainer.checkpoint_path)

    # model_scripted = torch.jit.script(model) # Export to TorchScript

    model.to(device)

    y, extra = run_inference_step(model, text_encoder, [args.text], device)
    wave = synth_audio(y, audio_frontend)
    torchaudio.save(args.output, wave, audio_config.sample_rate)

    from matplotlib import pyplot as plt

    plt.subplot(411)
    plt.imshow(y[0].detach().numpy().T, origin="lower")
    plt.subplot(412)
    plt.plot(wave[0])
    plt.grid()
    plt.subplot(413)
    plt.plot(extra["s"][0].detach().numpy())
    plt.grid()
    plt.subplot(414)
    plt.imshow(extra["w"][0].detach().numpy().T, origin="lower")
    plt.show()

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to synthesise")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("--checkpoint_dir", default="checkpoint_default", help="Checkpoint path")
    parser.add_argument("--output", default="output.wav", help="Output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
