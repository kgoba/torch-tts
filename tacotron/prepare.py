import yaml, os, sys, logging
from tqdm.auto import tqdm
import numpy as np
import torch

from data.dataset import build_dataset
from tacotron import lengths_to_mask
from train_util import find_available_device

logger = logging.getLogger(__name__)


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


def main(args):
    config = yaml.safe_load(open(args.config))

    dataset = build_dataset(args.dataset, config, args.data_path)
    # check_dataset_stats(audio_dataset)

    device = find_available_device()
    if args.cpu:
        device = "cpu"  # set PYTORCH_ENABLE_MPS_FALLBACK=1

    model = None

    for batch in tqdm(dataset):
        if model:
            c, c_lengths, x, x_lengths = [
                t.to(device) for t in batch if isinstance(t, torch.Tensor)
            ]

            xmask = lengths_to_mask(x_lengths).unsqueeze(2)

            y, y_post, s, out_dict = model(
                c, c_lengths, x, x_lengths, xref=x, xref_lengths=x_lengths
            )

    logger.info(f"Dataset size: {len(dataset)}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset path")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("--data", help="Data file path", dest="data_path")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU for training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
