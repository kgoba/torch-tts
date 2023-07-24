import torch
import numpy as np
import yaml, os, sys, logging
from tqdm.auto import tqdm

from train_util import find_available_device, Trainer
from dataset import collate_fn, build_dataset
from tacotron import build_tacotron

logger = logging.getLogger(__name__)


def loss_fn(x, y, mask):
    loss = torch.nn.functional.l1_loss(x, y, reduction="none")
    # loss = torch.nn.functional.mse_loss(x, y, reduction="none")
    loss = torch.mean(loss * mask, dim=2)
    loss = loss.sum() / mask.sum()
    return loss
    # return loss.sqrt()


def main(args):
    dataset_path = args.dataset
    config_path = args.config

    test_size = 200

    config = yaml.safe_load(open(config_path))
    random_seed = config["seed"] or 42

    dataset = build_dataset(dataset_path, config)
    # check_dataset_stats(audio_dataset)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    print(f"Dataset size: {len(train_dataset)} train + {len(test_dataset)} test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=64
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=16
    )

    device = find_available_device()
    print(f"Using device {device}")
    print(f"Number of CPUs: {os.cpu_count()}")

    # device = "cpu"  # set PYTORCH_ENABLE_MPS_FALLBACK=1

    model = build_tacotron(config)

    trainer = Trainer(model, args.checkpoint_dir)

    trainer.train(train_loader, test_loader, loss_fn, device)

    # model.to(device)
    # for batch in test_loader:
    #     input, imask, x, xmask = batch

    #     print(torch.min(x), torch.mean(x), torch.median(x), torch.max(x))

    #     x = x.to(device)
    #     xmask = xmask.to(device)

    #     y, w = model(input.to(device), imask.to(device), x)
    #     # print(x.shape, y.shape, y.dtype)
    #     print(x.device, y.device, xmask.device)

    #     loss = loss_fn(x, y, xmask)
    #     print(loss.shape, loss)
    #     loss.backward()
    #     break

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset path")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("--checkpoint_dir", default="checkpoint_default", help="Checkpoint path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
