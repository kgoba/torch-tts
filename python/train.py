import torch
import yaml, os, sys, logging
from tqdm.auto import tqdm

from train_util import find_available_device, Trainer
from data.dataset import collate_fn, build_dataset_hdf5
from tacotron import build_tacotron

logger = logging.getLogger(__name__)


def main(args):
    config_path = args.config

    test_size = 200

    config = yaml.safe_load(open(config_path))
    random_seed = config["seed"] or 42

    torch.random.manual_seed(142)

    dataset = build_dataset_hdf5(args.dataset, config)
    # dataset = build_dataset(args.dataset, config, args.data_path)
    # check_dataset_stats(audio_dataset)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    logger.info(f"Dataset size: {len(train_dataset)} train + {len(test_dataset)} test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=32
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=16
    )

    device = find_available_device()
    if args.cpu:
        device = "cpu"  # set PYTORCH_ENABLE_MPS_FALLBACK=1

    logger.info(f"Using device {device}")
    logger.info(f"Number of CPUs: {os.cpu_count()}")

    model = build_tacotron(config)

    if args.lr is not None:
        trainer = Trainer(model, args.run_dir, lr=float(args.lr))
    else:
        trainer = Trainer(model, args.run_dir)

    trainer.train(train_loader, test_loader, device)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset path")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("--run", default="run_default", help="Training run path", dest="run_dir")
    parser.add_argument("--data", help="Data file path", dest="data_path")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU for training")
    parser.add_argument("--lr", help="Override learning rate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
