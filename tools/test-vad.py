import torchaudio
import torch
import sys


def pad_audio(wave, sr, duration=0.1):
    silence = torch.zeros(int(sr * duration))
    wave = torch.concatenate([silence, wave, silence], dim=0)
    return wave


def trim_audio(wave, sr, keep_beginning: float = 0.1, keep_end: float = 0.1):
    wave = torchaudio.functional.vad(wave, sr, pre_trigger_time=keep_beginning)
    wave = torchaudio.functional.vad(wave.flip(-1), sr, pre_trigger_time=keep_end).flip(
        -1
    )
    return wave


def apply_fade(
    wave,
    sample_rate: int = 22050,
    fade_in_length: float = 0.1,
    fade_out_length: float = 0.1,
    fade_shape: str = "half_sine",
):
    fade_in_sample_length = int(sample_rate * fade_in_length)
    fade_out_sample_length = int(sample_rate * fade_out_length)
    transform = torchaudio.transforms.Fade(
        fade_in_sample_length, fade_out_sample_length, fade_shape
    )
    return transform(wave)


def apply_fade(wave, sr, duration=0.1):
    fade_len = int(sr * duration)
    win = torch.hann_window(2 * fade_len)
    wave[:fade_len] *= win[:fade_len]
    wave[-fade_len:] *= win[-fade_len:]
    return wave


print(f"Torchaudio version: {torchaudio.__version__}")

wave, sr = torchaudio.load(sys.argv[1])
wave = wave.squeeze(0)
wave_orig = wave
wave = apply_fade(wave, sr, duration=0.1)  # Apply smooth fade-in/out
wave = pad_audio(wave, sr, duration=0.2) # Add zero silence at start/end
wave = trim_audio(wave, sr, keep_beginning=0.1, keep_end=0.1) # Invoke torchaudio.vad
wave = apply_fade(wave, sr, duration=0.1)  # Apply smooth fade-in/out
print(f"Wave length: {wave.shape} (VAD) / {wave_orig.shape} (original)")
torchaudio.save(sys.argv[2], wave.unsqueeze(0), sr)


# from matplotlib import pyplot as plt
# import numpy as np

# def amp_to_log(x, eps=1e-6):
#     return 20 * np.log10(np.abs(x) + eps) * np.sign(x)

# plt.plot(amp_to_log(wave_orig.numpy()))
# plt.plot(amp_to_log(wave.numpy()))
# plt.grid()
# plt.show()
