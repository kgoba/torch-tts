import torch
import yaml, os, sys, logging

from train_util import find_available_device, Trainer
from data.dataset import collate_fn, build_dataset_hdf5
from data.sampler import LengthBucketRandomSampler, RandomBatchSampler
from tacotron import build_tacotron

# from tacotron_lightning import build_tacotron
# import pytorch_lightning as pl

logger = logging.getLogger(__name__)


def main(args):
    config_path = args.config

    test_size = 200

    config = yaml.safe_load(open(config_path))
    random_seed = config["seed"] or 42

    torch.random.manual_seed(142)

    dataset = build_dataset_hdf5(args.dataset, config, max_frames=args.max_audio_frames)
    # dataset = build_dataset(args.dataset, config, args.data_path)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    logger.info(f"Dataset size: {len(train_dataset)} train + {len(test_dataset)} test")

    bucket_size = 8 * args.batch_size
    train_batch_sampler = RandomBatchSampler(
        LengthBucketRandomSampler(
            train_dataset, bucket_size=bucket_size, len_fn=lambda x: len(x[3])
        ),
        args.batch_size,
        drop_last=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_sampler=train_batch_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=args.eval_batch_size
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

    trainer.train(train_loader, test_loader, device, optimizer_interval=args.opt_interval)

    # trainer = pl.Trainer(accelerator="mps", val_check_interval=100, limit_train_batches=100)
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset path")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("--run", default="run_default", help="Training run path", dest="run_dir")
    parser.add_argument("--data", help="Data file path", dest="data_path")
    parser.add_argument("--batch-size", help="Training batch size", type=int, default=32)
    parser.add_argument("--eval-batch-size", help="Training batch size", type=int, default=16)
    parser.add_argument(
        "--opt-interval", help="Optimizer accumulation interval", type=int, default=1
    )
    parser.add_argument("--max-audio-frames", help="Limit audio frames", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force using CPU for training")
    parser.add_argument("--lr", help="Override learning rate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
