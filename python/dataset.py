import torch
import torchaudio
import numpy as np
import h5py
import os


class TextEncoder:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.lookup = {c: i for i, c in enumerate(alphabet)}

    def encode(self, text):
        text = text.lower()
        encoded = [self.lookup[c] for c in text if c in self.lookup]
        return encoded
    
    def decode(self, enc):
        return [self.alphabet[i] for i in enc]


class TranscribedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, transcript_path, file_fn=None):
        if file_fn is None:
            file_fn = lambda x: x
        ds_dir = os.path.split(transcript_path)[0]
        with open(transcript_path) as text_file:
            lines_split = [line.strip().split('|')[:2] for line in text_file.readlines()]
            self.transcripts = [(file_id, transcript, os.path.join(ds_dir, file_fn(file_id))) for file_id, transcript in lines_split]
    
    def __len__(self):
        return len(self.transcripts)
    
    def __getitem__(self, index):
        utt_id, transcript, audio_path = self.transcripts[index]
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0) # mix multichannel to mono
        else:
            audio = audio.squeeze(dim=0)
        return utt_id, transcript, audio, sr


class TacotronDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dataset, audio_frontend, text_encoder, cache_path=None):
        self.ds = audio_dataset
        self.af = audio_frontend
        self.te = text_encoder
        self.fs = h5py.File(cache_path, 'a') if cache_path else None

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        utt_id, transcript, audio, sr = self.ds[index]
        if self.fs and utt_id in self.fs:
            D_db = self.fs.get(f'{utt_id}/stft')[()]
            M_db = self.fs.get(f'{utt_id}/mel')[()]
        else:
            D_db, M_db = self.af.encode(audio, sr)
            if self.fs:
                self.fs.create_dataset(f'{utt_id}/stft', data=D_db)
                self.fs.create_dataset(f'{utt_id}/mel', data=M_db)
        return utt_id, self.te.encode(transcript), D_db, M_db


def collate_fn_tacotron(x):
    lens = [len(M_db.T) for _, _, _, M_db in x]
    mask = [torch.from_numpy(np.ones((x, 1))) for x in lens]
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
    M_db = [torch.from_numpy(m_fwd(M_db).T) for _, _, _, M_db in x]
    M_db = torch.nn.utils.rnn.pad_sequence(M_db, batch_first=True)

    tlens = [len(input.T) for input, _, _, _ in x]
    tmask = [torch.from_numpy(np.ones((x, 1))) for x in tlens]
    tmask = torch.nn.utils.rnn.pad_sequence(tmask, batch_first=True)
    input = [torch.from_numpy(input.T) for input, _, _, _ in x]
    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)
    return [input, tmask, M_db, mask]


def collate_fn_melnet(x):
    lens = [len(M_db.T) for _, _, _, M_db in x]
    mask = [torch.from_numpy(np.ones((x, 1))) for x in lens]
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
    M_db = [torch.from_numpy(m_fwd(M_db).T) for _, _, _, M_db in x]
    M_db = torch.nn.utils.rnn.pad_sequence(M_db, batch_first=True)
    D_db = [torch.from_numpy(d_fwd(D_db).T) for _, _, D_db, _ in x]
    D_db = torch.nn.utils.rnn.pad_sequence(D_db, batch_first=True)
#     print('Collate', M_db.shape, D_db.shape, mask.shape)
    return [M_db, D_db, mask] # M_db, D_db, mask
