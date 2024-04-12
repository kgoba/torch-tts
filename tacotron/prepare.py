import yaml, sys, logging
from tqdm.auto import tqdm
import torch

from data.dataset import build_dataset
from tacotron import lengths_to_mask
from train_util import find_available_device

logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--cpu", action="store_true", help="Force using CPU for training"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
