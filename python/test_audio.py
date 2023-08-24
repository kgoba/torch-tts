import torch
import torchaudio
import yaml, os, sys, logging

from audio import AudioFrontend, AudioFrontendConfig

logger = logging.getLogger(__name__)

def main(args):
    config_path = args.config
    config = yaml.safe_load(open(config_path))

    audio_config = AudioFrontendConfig()
    audio_config.from_json(config["audio"])
    audio_frontend = AudioFrontend(audio_config)

    wave_in, sr = torchaudio.load(args.audio_in, channels_first=True)
    if wave_in.shape[0] > 1:
        wave_in = wave_in.mean(dim=0)  # mix multichannel to mono
    else:
        wave_in = wave_in.squeeze(dim=0)
    print(wave_in.shape)

    _, M_db = audio_frontend.encode(wave_in, sr)
    print(M_db.shape)

    D_db = audio_frontend.mel_inv(M_db)
    wave = audio_frontend.decode(D_db)
    wave = wave / wave.abs().max()

    if args.audio_out:
        torchaudio.save(args.audio_out, wave, audio_config.sample_rate)
    else:
        import sounddevice as sd
        print(wave.shape)
        sd.play(wave.numpy().T, audio_config.sample_rate)
        sd.wait()

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("audio_in", help="Input audio file")
    parser.add_argument("--audio_out", help="Output audio file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
