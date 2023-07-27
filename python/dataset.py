import torch
import torchaudio
import numpy as np
import h5py
import os, re, logging

logger = logging.getLogger(__name__)


class TextEncoder:
    def __init__(self, alphabet, char_map=None, bos=None, eos=None):
        self.char_map = dict(char_map) if char_map else dict()
        self.bos = bos
        self.eos = eos
        self.alphabet = alphabet
        self.lookup = {c: (i + 1) for i, c in enumerate(alphabet)}

    def encode(self, text, encode_unk=None):
        # text_orig = text
        text = text.lower()
        for key, value in self.char_map.items():
            text = re.sub(key, value, text)
        if self.bos:
            text = self.bos + text
        if self.eos:
            text = text + self.eof
        # if text != text_orig:
        #     logger.debug(f"Transformed [{text_orig}] to [{text}]")
        if encode_unk:
            encoded = [self.lookup[c] if c in self.lookup else encode_unk for c in text]
        else:
            encoded = [self.lookup[c] for c in text if c in self.lookup]
            unk_chars = ""
            for c in text:
                if not c in self.lookup:
                    unk_chars += c
            if unk_chars:
                logger.warning(f"Unknown characters: {unk_chars}")
        return encoded

    def decode(self, enc, decode_unk=None):
        if decode_unk:
            return [
                self.alphabet[i - 1] if i > 0 and i <= len(self.alphabet) else decode_unk
                for i in enc
            ]
        else:
            return [self.alphabet[i - 1] for i in enc if i > 0 and i <= len(self.alphabet)]


class TranscribedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, transcript_path, audio_dir, filename_fn=None, text_filter=None):
        if filename_fn is None:
            filename_fn = lambda x: x
        if text_filter is None:
            text_filter = lambda x: True
        with open(transcript_path) as text_file:
            lines_split = [line.strip().split("|")[:2] for line in text_file.readlines()]
            self.transcripts = [
                (
                    file_id,
                    transcript,
                    os.path.join(audio_dir, filename_fn(file_id)),
                )
                for file_id, transcript in lines_split
                if text_filter(transcript)
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
    def __init__(self, audio_dataset, audio_frontend, text_encoder, cache_path=None, r=1):
        self.ds = audio_dataset
        self.af = audio_frontend
        self.te = text_encoder
        self.fs = h5py.File(cache_path, "a") if cache_path else None
        self.r = r

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        utt_id, transcript, audio, sr = self.ds[index]
        if self.fs and utt_id in self.fs:
            # D_db = torch.from_numpy(self.fs.get(f"{utt_id}/stft")[()])
            D_db = None
            M_db = torch.from_numpy(self.fs.get(f"{utt_id}/mel")[()])
        else:
            D_db, M_db = self.af.encode(audio, sr)
            T = (D_db.shape[0] // self.r) * self.r
            D_db = D_db[:T, :]
            M_db = M_db[:T, :]
            assert D_db.shape[0] == M_db.shape[0]

            if self.fs:
                # self.fs.create_dataset(f"{utt_id}/stft", data=D_db)
                self.fs.create_dataset(f"{utt_id}/mel", data=M_db)
        return utt_id, self.te.encode(transcript), D_db, M_db


from audio import AudioFrontend, AudioFrontendConfig


def text_has_no_digits(text):
    return re.search(rf"\d", text) is None


def build_dataset(dataset_path, config, cache_path=None) -> TacotronDataset:
    # dataset_path = config["dataset"]["root"]
    audio_dataset = TranscribedAudioDataset(
        os.path.join(dataset_path, "transcripts.txt"),
        dataset_path,
        filename_fn=lambda x: x + ".wav",
        text_filter=text_has_no_digits,
    )

    audio_config = AudioFrontendConfig()
    audio_config.from_json(config["audio"])
    audio_frontend = AudioFrontend(audio_config)

    text_encoder = TextEncoder(config["text"]["alphabet"], config["text"].get("character_map"))

    return TacotronDataset(
        audio_dataset,
        audio_frontend,
        text_encoder,
        r=config["model"]["decoder"]["r"],
        cache_path=cache_path,
    )


def m_fwd(x):
    return (x + 120) / 120


def m_rev(x):
    return (x * 120) - 120


def collate_fn(data):
    M_db = [m_fwd(M_db) for _, _, _, M_db in data]
    omask = [torch.ones((len(x), 1), dtype=torch.bool) for x in M_db]
    omask = torch.nn.utils.rnn.pad_sequence(omask, batch_first=True)
    M_db = torch.nn.utils.rnn.pad_sequence(M_db, batch_first=True)

    input = [torch.LongTensor(input) for _, input, _, _ in data]
    imask = [torch.ones(len(x), dtype=torch.bool) for x in input]
    imask = torch.nn.utils.rnn.pad_sequence(imask, batch_first=True)
    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)

    return input, imask, M_db, omask


import tqdm


def check_dataset_stats(dataset, sample_size=500):
    print(f"Dataset size: {len(dataset)}")
    sample, _ = torch.utils.data.random_split(
        dataset,
        [sample_size, len(dataset) - sample_size],
        generator=torch.Generator().manual_seed(142),
    )
    utt_len = []
    audio_len = []
    audio_pwr = []
    for i in tqdm(range(len(sample))):
        utt_id, transcript, audio, sr = dataset[i]
        utt_len.append(len(transcript))
        audio_len.append(len(audio) / sr)
        audio_pwr.append(10 * np.log10(np.mean(audio.numpy() ** 2)))

    print(
        f"Utterance length: {np.median(utt_len):.1f} (median), {np.quantile(utt_len, 0.05):.1f}..{np.quantile(utt_len, 0.95):.1f} (5%..95%) characters"
    )
    print(
        f"Audio length:     {np.median(audio_len):.1f} (median), {np.quantile(audio_len, 0.05):.1f}..{np.quantile(audio_len, 0.95):.1f} (5%..95%) s"
    )
    print(
        f"Audio RMS power:  {np.median(audio_pwr):.1f} (median), {np.quantile(audio_pwr, 0.05):.1f}..{np.quantile(audio_pwr, 0.95):.1f} (5%..95%) dBFS"
    )
    print(f"Total audio length: {len(dataset) * np.mean(audio_len) / 3600:.1f} h (estimated)")
