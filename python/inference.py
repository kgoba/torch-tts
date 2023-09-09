import torch
import numpy as np
import torchaudio
import yaml, os, sys, logging

from train_util import find_available_device, Trainer
from data.dataset import m_rev, m_fwd, AudioFrontend, AudioFrontendConfig, TextEncoder
from tacotron import build_tacotron, run_inference_step

logger = logging.getLogger(__name__)


def synth_audio(y, audio_frontend):
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

    if args.ref:
        ref_audio, ref_sr = torchaudio.load(args.ref, channels_first=True)
        _, M_db = audio_frontend.encode(ref_audio, ref_sr)
        xref = m_fwd(M_db)
    else:
        xref = None

    model = build_tacotron(config)
    model.eval()

    trainer = Trainer(model, args.run_dir)
    trainer.load_checkpoint(trainer.checkpoint_path)

    # model_scripted = torch.jit.script(model) # Export to TorchScript

    model.to(device)

    y, extra = run_inference_step(
        model, text_encoder, [args.text], device, xref=xref, max_steps=args.max_steps
    )

    wave = synth_audio(y, audio_frontend)

    if args.output:
        torchaudio.save(args.output, wave, audio_config.sample_rate)

    if args.play:
        import sounddevice as sd

        sd.play(wave.numpy().T, audio_config.sample_rate)
        sd.wait()

    if args.plot:
        K = 9
        c = np.fft.rfft(y[0].numpy())
        c[:, K:] = 0
        yy = np.fft.irfft(c)[:, :80]
        c = c * (np.arange(41)[np.newaxis, ...])
        cc = np.concatenate([c[:, 1:K].T.real, c[:, 1:K].T.imag], axis=0)
        from matplotlib import pyplot as plt

        plt.subplot(311)
        plt.imshow(y[0].numpy().T, origin="lower")
        plt.subplot(312)
        w_0 = extra["w"][0].numpy()
        # w_0 = np.power(w_0, 2)
        # w_0 = w_0 / (1e-6 + w_0.sum(axis=1)[..., np.newaxis])
        # np.ndarray()
        # wt_0 = w_0/(1e-6 + w_0.sum(axis=0))
        # zz = np.dot(y[0].numpy().T, wt_0.repeat(2, axis=0))
        # yy = np.dot(w_0.repeat(2, axis=0), zz.T)
        plt.imshow(w_0.T, origin="lower")
        # plt.imshow(np.where(w_0 > 0.95, 1, 0).T, origin="lower")
        # plt.imshow(yy.T, origin="lower")
        # plt.imshow(cc, origin="lower", aspect=4)
        # plt.imshow(y[0].numpy().T - yy.T, origin="lower")
        # plt.imshow(yy.T, origin="lower")
        plt.subplot(313)
        # t = np.linspace(0, 1, w_0.shape[1])[np.newaxis, :]
        t = np.arange(w_0.shape[1])[np.newaxis, :]
        w_mean = np.sum(w_0 * t, axis=1)
        w2_mean = np.sum(w_0 * t**2, axis=1)
        w_var = w2_mean - w_mean**2
        plt.plot(np.sqrt(w_var))
        # plt.plot(extra["s"][0].numpy())
        plt.grid()
        plt.show()

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to synthesise")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument(
        "--run", default="checkpoint_default", help="Checkpoint path", dest="run_dir"
    )
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--ref", help="Reference audio")
    parser.add_argument("--play", action="store_true", help="Play audio")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--max-steps", type=int, help="Max decoder steps", default=400)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
