import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
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
            logger.warning(f"Did not set param {key}, skipping ({e})")


def loss_fn(x, y, mask):
    loss = torch.nn.functional.l1_loss(x, y, reduction="none")
    # loss = torch.nn.functional.mse_loss(x, y, reduction="none")
    loss = torch.mean(loss * mask, dim=2)
    loss = loss.sum() / mask.sum()
    return loss
    # return loss.sqrt()


def run_training_step(model, batch, device):
    input, imask, x, xmask = [x.to(device, non_blocking=True) for x in batch]
    y, s, w = model(input, imask, x)
    loss_mel = 120 * loss_fn(x, y, xmask)
    loss_stop = 10 * torch.nn.functional.binary_cross_entropy_with_logits(s, xmask.float())
    loss = loss_mel + loss_stop
    return loss, {
        "loss_mel": loss_mel.detach().item(),
        "loss_stop": loss_stop.detach().item(),
        "w": w.detach().cpu(),
    }


def loss_loop(model, batch_loader, device, num_steps=None, optimizer=None, scheduler=None):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    if num_steps:
        pbar = tqdm(batch_loader, total=num_steps)
    else:
        pbar = tqdm(batch_loader)

    loss_hist = []
    for idx_step, batch in enumerate(pbar):
        if optimizer is not None:
            optimizer.zero_grad()
        loss, loss_dict = run_training_step(model, batch, device)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        loss_hist.append(loss.item())

        loss_mel = loss_dict["loss_mel"]
        pbar.set_postfix_str(
            f"Loss: {loss.item():.3f} (mel: {loss_mel:.3f}), mean: {np.mean(loss_hist):.3f}"
        )
        if num_steps and idx_step + 1 >= num_steps:
            break

    return loss_hist, loss_dict["w"]


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
    def __init__(self, model: torch.nn.Module, checkpoint_dir: str, step: int = 0, lr=5e-4):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        self.step = step
        self.lr = lr

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, amsgrad=True, weight_decay=0.03
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        # Print model's state_dict
        logger.debug("Model's state_dict:")
        for param_tensor_name in model.state_dict():
            t = model.state_dict()[param_tensor_name]
            # print(
            #     f"{param_tensor_name}: \t{t.min():.3f} .. {t.max():.3f} avg {t.type(torch.float).mean():.3f} avg abs {t.type(torch.float).abs().mean():.3f}"
            # )
            logger.debug(param_tensor_name, "\t", t.size())

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of parameters: {total_params}")
        logger.info(f"Learning rate: {lr:.1e}")

    def save_checkpoint(self, path):
        logger.info(f"Saving checkpoint to {path}")
        checkpoint = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

        # try:
        #     scripted_path = os.path.join(self.checkpoint_dir, f"model_scripted.pt")
        #     model_scripted = torch.jit.script(self.model)  # Export to TorchScript
        #     model_scripted.save(scripted_path)  # Save
        # except Exception as e:
        #     logger.warning(f"Failed to save scripted model ({e})")

    def load_checkpoint(self, path):
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        self.step = checkpoint["step"]
        try:
            self.model.load_state_dict(checkpoint["model_state"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr
        except:
            logger.warning("Failed to load model properly, attempting to load partially")
            load_state_dict(self.model, checkpoint["model_state"])
            logger.warning("Optimizer state not restored")
        logger.info(f"Training steps: {self.step}")

    def train(self, train_loader, test_loader, device, num_epochs=600):
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)
        pmodel = self.model if device == "cpu" else torch.nn.DataParallel(self.model)
        pmodel.to(device)

        optimizer_to(self.optimizer, device)

        loss_hist = []
        for epoch_idx in range(num_epochs):
            epoch_loss, _ = loss_loop(
                pmodel, train_loader, device, optimizer=self.optimizer, num_steps=100
            )
            self.step += len(epoch_loss)

            epoch_test_loss, w_test = loss_loop(pmodel, test_loader, device, num_steps=10)
            loss_hist.append([np.mean(epoch_loss), np.mean(epoch_test_loss)])

            self.save_checkpoint(self.checkpoint_path)
            # if self.step % 1000 == 0:
            #     model_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.step}.pt")
            #     self.save_checkpoint(model_path)

            alignment_path = os.path.join(self.checkpoint_dir, f"alignment_{self.step}.png")
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            axes.imshow(w_test[0].numpy().T, origin="lower")
            axes.set_title(f"Step: {self.step} | Train loss: {np.mean(epoch_loss):.2f}")
            fig.tight_layout()
            fig.savefig(alignment_path, bbox_inches="tight")
            plt.close(fig)
