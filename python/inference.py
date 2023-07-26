import torch
import numpy as np
import yaml, os, sys, logging

from train_util import find_available_device, Trainer
from dataset import collate_fn, build_dataset, TextEncoder
from tacotron import build_tacotron

logger = logging.getLogger(__name__)


def main(args):
    config_path = args.config

    config = yaml.safe_load(open(config_path))
    random_seed = config["seed"] or 42

    device = find_available_device()
    print(f"Using device {device}")
    print(f"Number of CPUs: {os.cpu_count()}")

    device = "cpu"  # set PYTORCH_ENABLE_MPS_FALLBACK=1

    text_encoder = TextEncoder(config["text"]["alphabet"], config["text"].get("character_map"))

    model = build_tacotron(config)
    model.eval()

    trainer = Trainer(model, args.checkpoint_dir)
    trainer.load_checkpoint(trainer.checkpoint_path)

    # model_scripted = torch.jit.script(model) # Export to TorchScript

    model.to(device)
    input = torch.Tensor([text_encoder.encode(args.text)]).to(dtype=torch.long, device=device)
    imask = torch.ones_like(input).to(torch.bool)
    print(input, imask)
    y, s, w = model(input, imask)

    from matplotlib import pyplot as plt
    plt.subplot(311)
    plt.imshow(w[0].detach().numpy().T, origin='lower')
    plt.subplot(312)
    plt.imshow(y[0].detach().numpy().T, origin='lower')
    plt.subplot(313)
    plt.plot(s[0].detach().numpy())
    plt.grid()
    plt.show()

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to synthesise")
    parser.add_argument("config", help="Configuration file")
    parser.add_argument("--checkpoint_dir", default="checkpoint_default", help="Checkpoint path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    rc = main(args)
    sys.exit(rc)
