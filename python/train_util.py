import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import logging

logger = logging.getLogger(__name__)


def find_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def load_state_dict(model, state_dict):
    # https://github.com/pytorch/pytorch/issues/40859
    def get_attr(obj, names):
        if len(names) == 1:
            return getattr(obj, names[0])
        else:
            return get_attr(getattr(obj, names[0]), names[1:])

    def set_attr(obj, names, val):
        if len(names) == 1:
            setattr(obj, names[0], val)
        else:
            set_attr(getattr(obj, names[0]), names[1:], val)

    for key, dict_param in state_dict.items():
        submod_names = key.split(".")
        try:
            curr_param = get_attr(model, submod_names)
            new_param = dict_param  #  torch.nn.Parameter(dict_param)
            with torch.no_grad():
                curr_param.data.copy_(new_param)
        except Exception as e:
            print(f"Did not set param {key}, skipping ({e})")


def loss_loop(model, loss_fn, batch_loader, device, num_steps=None, optimizer=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    if num_steps:
        pbar = tqdm(enumerate(batch_loader), total=num_steps)
    else:
        pbar = tqdm(enumerate(batch_loader))

    loss_hist = []
    for batch_idx, batch in pbar:
        input, imask, x, xmask = [x.to(device, non_blocking=True) for x in batch]
        if optimizer is not None:
            optimizer.zero_grad()
        y, s, w = model(input, imask, x)
        loss_mel = 120 * loss_fn(x, y, xmask)
        loss_stop = 10 * torch.nn.functional.binary_cross_entropy_with_logits(s, xmask.float())
        loss = loss_mel + loss_stop
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        loss_hist.append(loss.item())
        pbar.set_postfix_str(
            f"Loss: {loss.item():.3f} (mel: {loss_mel.item():.3f}), mean: {np.mean(loss_hist):.3f}"
        )
        if num_steps and batch_idx + 1 >= num_steps:
            break
    # print(imask[0].to(torch.int).sum(), xmask[0].to(torch.int).sum())
    # print((xmask[:, ::3, :] * imask.unsqueeze(1)).shape)
    # print(imask.shape, xmask.shape, w.shape)
    return loss_hist, w


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class Trainer:
    def __init__(self, model: torch.nn.Module, checkpoint_dir: str, step: int = 0):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        self.step = step

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor_name in model.state_dict():
            t = model.state_dict()[param_tensor_name]
            # print(
            #     f"{param_tensor_name}: \t{t.min():.3f} .. {t.max():.3f} avg {t.type(torch.float).mean():.3f} avg abs {t.type(torch.float).abs().mean():.3f}"
            # )
            print(param_tensor_name, "\t", t.size())

        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Number of parameters: {total_params}")

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-4, amsgrad=True, weight_decay=0.01
        )

    def save_checkpoint(self, path):
        logger.info(f"Saving checkpoint to {path}")
        checkpoint = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        # model2.to('cpu')
        # model_scripted = torch.jit.script(model) # Export to TorchScript
        # model_scripted.save('/kaggle/working/model_scripted.pt') # Save

    def load_checkpoint(self, path):
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        self.step = checkpoint["step"]
        # load_state_dict(self.model, checkpoint["model_state"])
        self.model.load_state_dict(checkpoint["model_state"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info(f"Training steps: {self.step}")

    def train(self, train_loader, test_loader, loss_fn, device, num_epochs=600):
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)
        pmodel = self.model if device == "cpu" else torch.nn.DataParallel(self.model)
        pmodel.to(device)

        optimizer_to(self.optimizer, device)

        loss_hist = []
        for epoch_idx in range(num_epochs):
            epoch_loss, _ = loss_loop(
                pmodel, loss_fn, train_loader, device, optimizer=self.optimizer, num_steps=100
            )
            self.step += len(epoch_loss)

            epoch_test_loss, w_test = loss_loop(pmodel, loss_fn, test_loader, device, num_steps=10)
            loss_hist.append([np.mean(epoch_loss), np.mean(epoch_test_loss)])

            self.save_checkpoint(self.checkpoint_path)

            alignment_path = os.path.join(self.checkpoint_dir, f"alignment_{self.step}.png")
            plt.imshow(w_test[0].detach().cpu().numpy())
            plt.savefig(alignment_path)
