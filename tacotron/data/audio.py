from dataclasses import dataclass
from torchaudio.functional import amplitude_to_DB, DB_to_amplitude, resample, vad
from torchaudio.transforms import MelScale, InverseMelScale, Spectrogram, GriffinLim

import logging


@dataclass
class AudioFrontendConfig:
    sample_rate: int = 16000
    hop_length: int = 256
    win_length: int = 768
    num_mels: int = 80
    fmin: int = 50
    fmax: int = 7600

    def from_json(self, json):
        for key in json:
            self.__setattr__(key, json[key])
        return self


class AudioFrontend:
    def __init__(self, config):
        self.config = config
        logging.info(f"AudioFrontend config: {str(config)}")

        len_fft = config.win_length
        hop_length = config.hop_length
        n_freqs = (len_fft // 2) + 1

        self.stft_to_mels = MelScale(
            n_mels=self.config.num_mels,
            sample_rate=self.config.sample_rate,
            n_stft=n_freqs,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            norm="slaney",
        )
        self.mels_to_stft = InverseMelScale(
            n_mels=self.config.num_mels,
            sample_rate=self.config.sample_rate,
            n_stft=n_freqs,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            norm="slaney",
        )
        self.spectrogram = Spectrogram(
            n_fft=len_fft, hop_length=hop_length, power=2, normalized=True, center=True
        )
        self.griffinlim = GriffinLim(n_fft=len_fft, hop_length=hop_length, power=2)

    def encode(self, wave, sr):
        if sr != self.config.sample_rate:
            wave = resample(wave, orig_freq=sr, new_freq=self.config.sample_rate)
            sr = self.config.sample_rate
        wave = wave / wave.abs().max()
        # wave = vad(wave, sr, pre_trigger_time=0.05)
        # wave = vad(wave.flip(0), sr, pre_trigger_time=0.1).flip(0)
        # wave = wave / wave.abs().max()
        D = self.spectrogram(wave)
        M = self.stft_to_mels(D)
        D_db = amplitude_to_DB(D, 10, 1e-12, 0)
        M_db = amplitude_to_DB(M, 10, 1e-12, 0)
        return D_db.mT, M_db.mT

    def decode(self, D_db):
        D = DB_to_amplitude(D_db, 1, 1)
        return self.griffinlim(D)

    def mel_inv(self, M_db):
        M = DB_to_amplitude(M_db.mT, 1, 1)
        D = self.mels_to_stft(M)
        return amplitude_to_DB(D, 10, 1e-12, 0)
