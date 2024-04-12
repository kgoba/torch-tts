import torch
import yaml, os, sys, logging
import lightning as L

from data.dataset import collate_fn, build_dataset_hdf5
from data.sampler import LengthBucketRandomSampler, RandomBatchSampler
from tacotron import build_tacotron
from tacotron_lightning import TacotronTask
from train_util import find_available_device, Trainer


def train(args, model, dataset, device):
    test_size = args.eval_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    logging.info(f"Dataset size: {len(train_dataset)} train + {len(test_dataset)} test")

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
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=args.eval_batch_size,
        num_workers=2,
        pin_memory=True,
    )

    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.enabled = True

    if args.lightning:
        task = TacotronTask(model, lr=args.lr, extra_loss=args.finetune)
        logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir="lightning_logs", version=0
        )
        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath="checkpoints",
            save_last=True,
            # every_n_train_steps=10,
            every_n_epochs=1,
        )
        trainer = L.Trainer(
            logger=logger,
            callbacks=[checkpoint_callback],
            accelerator="cpu" if args.cpu else "auto",
            accumulate_grad_batches=args.opt_interval,
            precision=args.precision,
            # val_check_interval=100,
            # limit_train_batches=100,
        )
        trainer.fit(
            task,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
            ckpt_path="last",
        )
    else:
        trainer = Trainer(model, args.run_dir, lr=args.lr)
        trainer.train(
            train_loader, test_loader, device, optimizer_interval=args.opt_interval
        )


def filter(args, model, dataset, device):
    from tacotron import run_training_step

    model.to(device)
    model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=collate_fn, batch_size=1  # args.eval_batch_size
    )

    with torch.no_grad():
        for batch in data_loader:
            loss, d = run_training_step(model, batch, device)
            for utt_id, T_i, w_i in zip(batch[0], batch[4], d["w"]):  # [T, L]
                w_max, _ = w_i.max(dim=1)  # [T]
                # crispness = w_max.mean()
                steps_i = T_i // 2
                crispness = torch.sum((w_max > 0.95).float()) / steps_i
                # print(f"{utt_id}\t{crispness.item()}")
                print(f"{utt_id}\t{steps_i}\t{w_max.mean().item()}\t{crispness}")
                # print(utt_id)


def main(args):
    config_path = args.config

    config = yaml.safe_load(open(config_path))

    random_seed = config["seed"] or 42
    torch.random.manual_seed(random_seed)

    dataset = build_dataset_hdf5(args.dataset, config, max_frames=args.max_audio_frames)
    # dataset = build_dataset(args.dataset, config, args.data_path)

    device = find_available_device()
    if args.cpu:
        device = "cpu"  # set PYTORCH_ENABLE_MPS_FALLBACK=1

    logging.info(f"Using device {device}")
    logging.info(f"Number of CPUs: {os.cpu_count()}")

    model = build_tacotron(config)

    train(args, model, dataset, device)
    # filter(args, model, dataset, device)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset path")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument(
        "--run", default="run_default", help="Training run path", dest="run_dir"
    )
    parser.add_argument("--data", help="Data file path", dest="data_path")
    parser.add_argument("--precision", help="Training precision", default="32")
    parser.add_argument(
        "--batch-size", help="Training batch size", type=int, default=32
    )
    parser.add_argument(
        "--eval-batch-size", help="Validation batch size", type=int, default=20
    )
    parser.add_argument(
        "--eval-size", help="Validation dataset size", type=int, default=80
    )
    parser.add_argument(
        "--opt-interval", help="Optimizer accumulation interval", type=int, default=1
    )
    parser.add_argument(
        "--max-audio-frames", help="Limit audio frames", type=int, default=None
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force using CPU for training"
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "--lightning", action="store_true", help="Use Lightning for training"
    )
    parser.add_argument(
        "--finetune", action="store_true", help="Finetuning with extra loss"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
