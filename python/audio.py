import numpy as np
import torchaudio


class AudioFrontendConfig:
    sample_rate = 16000
    hop_length = 0.010
    win_length = 0.030
    num_mels = 80
    fmin = 50
    fmax = 7600

    def from_json(self, json):
        for key in json:
            self.__setattr__(key, json[key])
        return self


class AudioFrontend:
    def __init__(self, config):
        self.config = config
        self.n_fft = int(0.5 + config.sample_rate * config.win_length)
        self.hop_length = int(0.5 + config.sample_rate * config.hop_length)
        n_stft = (self.n_fft // 2) + 1
        self.stft_to_mels = torchaudio.transforms.MelScale(
            n_mels=self.config.num_mels,
            sample_rate=self.config.sample_rate,
            n_stft=n_stft,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            norm="slaney",
        )
        self.mels_to_stft = torchaudio.transforms.InverseMelScale(
            n_mels=self.config.num_mels,
            sample_rate=self.config.sample_rate,
            n_stft=n_stft,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            norm="slaney",
        )

    def encode(self, wave, sr):
        if sr != self.config.sample_rate:
            wave = torchaudio.functional.resample(
                wave, orig_freq=sr, new_freq=self.config.sample_rate
            )
        D = torchaudio.functional.spectrogram(
            wave, n_fft=self.n_fft, hop_length=self.hop_length, power=1
        )
        M = self.stft_to_mels(D)
        D_db = torchaudio.functional.amplitude_to_DB(D)
        M_db = torchaudio.functional.amplitude_to_DB(M)
        return D_db, M_db

    def decode(self, D_db):
        D = torchaudio.functional.DB_to_amplitude(D_db)
        return (
            torchaudio.functional.griffinlim(D, hop_length=self.hop_length, power=1, momentum=0.9),
            self.config.sample_rate,
        )

    def mel_inv(self, M_db):
        M = torchaudio.functional.DB_to_amplitude(M_db)
        D = self.mels_to_stft(M)
        return torchaudio.functional.amplitude_to_DB(D)
