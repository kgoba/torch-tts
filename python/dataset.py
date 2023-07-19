import torch
import torchaudio
import numpy as np
import h5py
import os, re, logging

logger = logging.getLogger(__name__)


class TextEncoder:
    def __init__(self, alphabet, char_map=None):
        self.char_map = char_map or dict()
        self.alphabet = alphabet
        self.lookup = {c: i for i, c in enumerate(alphabet)}

    def encode(self, text):
        # text_orig = text
        text = text.lower()
        for key, value in self.char_map:
            text = re.sub(key, value, text)
        # logger.debug(f"Transformed [{text_orig}] to [{text}]")
        encoded = [self.lookup[c] if c in self.lookup else -1 for c in text]
        return encoded

    def decode(self, enc, decode_unk=None):
        if decode_unk:
            return [
                self.alphabet[i] if i >= 0 and i < len(self.alphabet) else decode_unk for i in enc
            ]
        else:
            return [self.alphabet[i] for i in enc if i >= 0 and i < len(self.alphabet)]


class TranscribedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, transcript_path, audio_dir, filename_fn=None):
        if filename_fn is None:
            filename_fn = lambda x: x
        with open(transcript_path) as text_file:
            lines_split = [line.strip().split("|")[:2] for line in text_file.readlines()]
            self.transcripts = [
                (file_id, transcript, os.path.join(audio_dir, filename_fn(file_id)))
                for file_id, transcript in lines_split
            ]

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, index):
        utt_id, transcript, audio_path = self.transcripts[index]
        audio, sr = torchaudio.load(audio_path, channels_first=True)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)  # mix multichannel to mono
        else:
            audio = audio.squeeze(dim=0)
        return utt_id, transcript, audio, sr


class TacotronDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dataset, audio_frontend, text_encoder, cache_path=None):
        self.ds = audio_dataset
        self.af = audio_frontend
        self.te = text_encoder
        self.fs = h5py.File(cache_path, "a") if cache_path else None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        utt_id, transcript, audio, sr = self.ds[index]
        if self.fs and utt_id in self.fs:
            D_db = self.fs.get(f"{utt_id}/stft")[()]
            M_db = self.fs.get(f"{utt_id}/mel")[()]
        else:
            D_db, M_db = self.af.encode(audio, sr)
            if self.fs:
                self.fs.create_dataset(f"{utt_id}/stft", data=D_db)
                self.fs.create_dataset(f"{utt_id}/mel", data=M_db)
        return utt_id, self.te.encode(transcript), D_db, M_db


from audio import AudioFrontend, AudioFrontendConfig


def build_dataset(dataset_path, config) -> TacotronDataset:
    # dataset_path = config["dataset"]["root"]
    audio_dataset = TranscribedAudioDataset(
        os.path.join(dataset_path, "transcripts.txt"),
        dataset_path,
        filename_fn=lambda x: x + ".wav",
    )

    audio_config = AudioFrontendConfig()
    audio_config.from_json(config["audio"])
    audio_frontend = AudioFrontend(audio_config)

    text_encoder = TextEncoder(config["text"]["alphabet"])

    return TacotronDataset(audio_dataset, audio_frontend, text_encoder)


def m_fwd(x):
    return (x + 120) / 120


def m_rev(x):
    return (x * 120) - 120


def collate_fn_tacotron(data):
    M_db = [m_fwd(M_db).T for _, _, _, M_db in data]
    omask = [torch.ones((len(x), 1), dtype=torch.int) for x in M_db]
    omask = torch.nn.utils.rnn.pad_sequence(omask, batch_first=True)
    M_db = torch.nn.utils.rnn.pad_sequence(M_db, batch_first=True)

    input = [torch.IntTensor(input) for _, input, _, _ in data]
    imask = [torch.ones(len(x), dtype=torch.int) for x in input]
    imask = torch.nn.utils.rnn.pad_sequence(imask, batch_first=True)
    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)

    return input, imask, M_db, omask


def collate_fn_melnet(x):
    lens = [len(M_db.T) for _, _, _, M_db in x]
    mask = [torch.from_numpy(np.ones((x, 1))) for x in lens]
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
    M_db = [torch.from_numpy(m_fwd(M_db).T) for _, _, _, M_db in x]
    M_db = torch.nn.utils.rnn.pad_sequence(M_db, batch_first=True)
    D_db = [torch.from_numpy(d_fwd(D_db).T) for _, _, D_db, _ in x]
    D_db = torch.nn.utils.rnn.pad_sequence(D_db, batch_first=True)
    #     print('Collate', M_db.shape, D_db.shape, mask.shape)
    return [M_db, D_db, mask]  # M_db, D_db, mask
