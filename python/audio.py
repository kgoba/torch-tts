from torchaudio.functional import (
    amplitude_to_DB,
    DB_to_amplitude,
    griffinlim,
    resample,
)
from torchaudio.transforms import MelScale, InverseMelScale, Spectrogram


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
        self.stft_to_mels = MelScale(
            n_mels=self.config.num_mels,
            sample_rate=self.config.sample_rate,
            n_stft=n_stft,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            norm="slaney",
        )
        self.mels_to_stft = InverseMelScale(
            n_mels=self.config.num_mels,
            sample_rate=self.config.sample_rate,
            n_stft=n_stft,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            norm="slaney",
        )
        self.spectrogram = Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=2, normalized=True)

    # def spectrogram(self, wave):
    #     x = wave # .numpy()
    #     nperseg=self.n_fft
    #     _, _, z = scipy.signal.stft(x, nfft=self.n_fft, nperseg=nperseg, noverlap=nperseg-self.hop_length, scaling='psd')
    #     z_mag2 = z.imag**2 + z.real**2
    #     return torch.as_tensor(z_mag2)

    def encode(self, wave, sr):
        if sr != self.config.sample_rate:
            wave = resample(wave, orig_freq=sr, new_freq=self.config.sample_rate)
        D = self.spectrogram(wave)
        # D = D_cpx.real**2 + D_cpx.imag**2
        M = self.stft_to_mels(D)
        D_db = amplitude_to_DB(D, 10, 1e-12, 0)
        M_db = amplitude_to_DB(M, 10, 1e-12, 0)
        return D_db.T, M_db.T

    def decode(self, D_db):
        D = DB_to_amplitude(D_db)
        return (
            griffinlim(D, hop_length=self.hop_length, power=1, momentum=0.9),
            self.config.sample_rate,
        )

    def mel_inv(self, M_db):
        M = DB_to_amplitude(M_db)
        D = self.mels_to_stft(M)
        return amplitude_to_DB(D)
