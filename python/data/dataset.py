import torch
import torchaudio
import numpy as np
import h5py
import os, re, logging

from data.text import TextEncoder, text_has_no_digits
from data.audio import AudioFrontend, AudioFrontendConfig

logger = logging.getLogger(__name__)


class TranscribedAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        transcript_path,
        audio_dir,
        utt_id_fn=None,
        utt_path_fn=None,
        text_filter=None,
        id_column=0,
        text_column=1,
    ):
        if utt_path_fn is None:
            utt_path_fn = lambda x: x
        if utt_id_fn is None:
            utt_id_fn = lambda x: x
        if text_filter is None:
            text_filter = lambda x: True

        with open(transcript_path) as text_file:
            lines_split = [line.strip().split("|") for line in text_file.readlines()]
            self.transcripts = [
                (
                    utt_id_fn(raw_id),
                    transcript,
                    os.path.join(audio_dir, utt_path_fn(raw_id)),
                )
                for raw_id, transcript in [
                    (entry[id_column], entry[text_column]) for entry in lines_split
                ]
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
            # D_db = torch.from_numpy(self.fs.get(f"{utt_id}/stft")[()])
            D_db = None
            M_db = torch.from_numpy(self.fs.get(f"{utt_id}/mel")[()])
            if "text" not in self.fs[utt_id]:
                print(f"Saving [{utt_id}]={transcript}")
                self.fs.create_dataset(f"{utt_id}/text", data=transcript)
        else:
            D_db, M_db = self.af.encode(audio, sr)
            assert D_db.shape[0] == M_db.shape[0]

            if self.fs:
                # self.fs.create_dataset(f"{utt_id}/stft", data=D_db)
                self.fs.create_dataset(f"{utt_id}/mel", data=M_db)
                self.fs.create_dataset(f"{utt_id}/text", data=transcript)
        return utt_id, self.te.encode(transcript), D_db, M_db


class TacotronDatasetHDF5(torch.utils.data.Dataset):
    def __init__(self, data_path, text_encoder):
        self.te = text_encoder
        self.fs = h5py.File(data_path, "r")
        self.utt_ids = list(self.fs.keys())

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]
        D_db = None
        M_db = torch.from_numpy(self.fs.get(f"{utt_id}/mel")[()])
        transcript = self.fs.get(f"{utt_id}/text").asstr()[()]
        return utt_id, self.te.encode(transcript), D_db, M_db


def regex_replace_fn(x, re_match, re_replace):
    y = re.sub(f"^{re_match}$", re_replace, x)
    return y


def build_dataset_hdf5(dataset_path, config) -> TacotronDatasetHDF5:
    text_config = config["text"]
    text_encoder = TextEncoder(
        text_config["alphabet"],
        text_config.get("character_map"),
        bos=text_config.get("bos_symbols"),
        eos=text_config.get("eos_symbols"),
    )
    return TacotronDatasetHDF5(
        dataset_path,
        text_encoder,
    )


def build_dataset(dataset_path, config, cache_path=None) -> TacotronDataset:
    dataset_config = config["dataset"]

    utt_path_fn = lambda x: regex_replace_fn(
        x, dataset_config["utt_id"]["re_match"], dataset_config["utt_id"]["re_path"]
    )
    utt_id_fn = lambda x: regex_replace_fn(
        x, dataset_config["utt_id"]["re_match"], dataset_config["utt_id"]["re_id"]
    )

    audio_dataset = TranscribedAudioDataset(
        os.path.join(dataset_path, dataset_config["transcript"]),
        dataset_path,
        utt_path_fn=utt_path_fn,
        utt_id_fn=utt_id_fn,
        text_filter=text_has_no_digits,
        id_column=dataset_config["utt_id"]["column"],
        text_column=dataset_config["utt_text"]["column"],
    )

    audio_config = AudioFrontendConfig()
    audio_config.from_json(config["audio"])
    audio_frontend = AudioFrontend(audio_config)

    text_config = config["text"]
    text_encoder = TextEncoder(
        text_config["alphabet"],
        text_config.get("character_map"),
        bos=text_config.get("bos_symbols"),
        eos=text_config.get("eos_symbols"),
    )

    return TacotronDataset(
        audio_dataset,
        audio_frontend,
        text_encoder,
        cache_path=cache_path,
    )


def m_fwd(x):
    return (x + 120) / 120


def m_rev(x):
    return (x * 120) - 120


def collate_fn(data):
    # ids = [utt_id for utt_id, _, _, _ in data]
    # print(ids)

    M_db = [m_fwd(M_db) for _, _, _, M_db in data]
    output_lengths = torch.LongTensor([len(x) for x in M_db])
    M_db = torch.nn.utils.rnn.pad_sequence(M_db, batch_first=True)

    input = [torch.LongTensor(input) for _, input, _, _ in data]
    input_lengths = torch.LongTensor([len(x) for x in input])
    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True)

    return input, input_lengths, M_db, output_lengths


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
