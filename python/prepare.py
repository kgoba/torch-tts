import yaml, os, sys, logging
from tqdm.auto import tqdm

from data.dataset import build_dataset

logger = logging.getLogger(__name__)


def main(args):
    dataset_path = args.dataset
    config_path = args.config

    config = yaml.safe_load(open(config_path))

    dataset = build_dataset(dataset_path, config, args.data_path)
    # check_dataset_stats(audio_dataset)

    for _ in tqdm(dataset):
        pass

    logger.info(f"Dataset size: {len(dataset)}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset path")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("--data", help="Data file path", dest="data_path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
