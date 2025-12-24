from typing import Sequence, overload
import numpy as np
import os, glob, sys, random, logging
import soundfile
from tqdm.auto import tqdm


class WavDataset(Sequence):
    def __init__(self, audio_dir):
        self.audio_paths = glob.glob(os.path.join(audio_dir, "*.wav"))

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        with soundfile.SoundFile(self.audio_paths[index]) as f:
            sr = f.samplerate
            data = f.read(always_2d=True, dtype="float32")  # [frames, channels]
        return data, sr


def as_windows(x, length, stride=1, axis=0):
    for start in range(0, x.shape[axis] - length, stride):
        yield x.take(indices=range(start, start+length), axis=axis)


def check_dataset_stats(dataset, sample_size=None, win_length=768, hop_length=256):
    print(f"Dataset size: {len(dataset)}, sample size: {sample_size}")
    if sample_size:
        sample_idx = random.sample(range(len(dataset)), sample_size)
        sample = [dataset[i] for i in sample_idx]
    else:
        sample = dataset

    utt_len = []
    audio_len = []
    audio_pwr = []
    audio_pwr_start = []
    for audio, sr in tqdm(sample):
        # utt_len.append(len(transcript))
        audio = np.mean(audio, axis=1)

        audio_len.append(len(audio) / sr)
        audio_pwr.append(10 * np.log10(np.mean(audio**2)))
        # frames = list(as_windows(audio**2, win_length, hop_length))
        # audio_pwr_start.append(10 * np.log10(np.mean(frames[:50], axis=1)))

    # print(
    #     f"Utterance length: {np.median(utt_len):.1f} (median), {np.quantile(utt_len, 0.05):.1f}..{np.quantile(utt_len, 0.95):.1f} (5%..95%) characters"
    # )
    print(
        f"Audio length:     {np.median(audio_len):.1f} (median), {np.quantile(audio_len, 0.05):.1f}..{np.quantile(audio_len, 0.95):.1f} (5%..95%) s"
    )
    print(
        f"Audio RMS power:  {np.median(audio_pwr):.1f} (median), {np.quantile(audio_pwr, 0.05):.1f}..{np.quantile(audio_pwr, 0.95):.1f} (5%..95%) dBFS"
    )
    print(f"Total audio length: {len(dataset) * np.mean(audio_len) / 3600:.1f} h (estimated)")

    import matplotlib.pyplot as plt

    # plt.hist(audio_pwr, bins=np.linspace(-30, -10, 40), density=True)
    plt.hist(audio_len, bins=25, density=True)
    # plt.plot(np.quantile(audio_pwr_start, 0.20, axis=0))
    # plt.plot(np.mean(audio_pwr_start, axis=0))
    # plt.plot(np.quantile(audio_pwr_start, 0.80, axis=0))
    plt.grid()
    plt.show()


def main(args):
    dataset = WavDataset(args.audio_dir)
    check_dataset_stats(dataset, args.sample)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", help="Audio file directory")
    parser.add_argument("--sample", help="Sample size", type=int)
    args = parser.parse_args()

    # logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
