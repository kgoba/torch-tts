import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import os
import logging
import gc

from tacotron import run_training_step

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


def grad_norm(model):
    grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad != None]
    if grads:
        return torch.cat(grads).norm()


def loss_loop(
    model,
    batch_loader,
    device,
    num_steps=None,
    optimizer=None,
    scheduler=None,
    optimizer_interval=1,
    max_grad_norm=1,
    # step_callback=None
):
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    if num_steps:
        pbar = tqdm(batch_loader, total=num_steps)
    else:
        pbar = tqdm(batch_loader)

    loss_hist = []
    gn = 0
    for idx_step, batch in enumerate(pbar):
        if optimizer != None and (idx_step % optimizer_interval) == 0:
            optimizer.zero_grad()

        loss, loss_dict = run_training_step(model, batch, device)

        if optimizer != None:
            (loss / optimizer_interval).backward()
            gn = grad_norm(model)
            if ((idx_step + 1) % optimizer_interval) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler != None:
                    scheduler.step()

        # step_callback(loss.item(), loss_dict)
        loss_w = loss_dict["loss_w"]
        loss_kl = loss_dict["loss_kl"]
        loss_mel = loss_dict["loss_mel_db"]
        loss_mel_post = loss_dict["loss_mel_post_db"]
        loss_hist.append(loss_mel_post)

        pbar.set_postfix_str(
            f"GN: {gn:.3f} L: {loss.item():.4f} (mel: {loss_mel:.3f}/{loss_mel_post:.3f}/{np.mean(loss_hist):.3f}), w: {loss_w:.3f}, kl: {loss_kl:.3f}"
        )
        if num_steps and idx_step + 1 >= num_steps:
            break

    if optimizer != None:
        optimizer.zero_grad()

    return loss_hist, loss_dict


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

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr) # , amsgrad=True)
        # self.optimizer = torch.optim.NAdam(model.parameters(), lr=self.lr, momentum_decay=0)
        # self.optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr)
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
            # assert False
            self.model.load_state_dict(checkpoint["model_state"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr
        except:
            logger.warning("Failed to load model properly, attempting to load partially")
            load_state_dict(self.model, checkpoint["model_state"])
            logger.warning("Optimizer state not restored")
        logger.info(f"Training steps: {self.step}")

    def train(self, train_loader, test_loader, device, num_epochs=600, optimizer_interval=1):
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint(self.checkpoint_path)

        pmodel = self.model if device == "cpu" else torch.nn.DataParallel(self.model)
        pmodel.to(device)

        optimizer_to(self.optimizer, device)

        loss_hist = []
        for epoch_idx in range(num_epochs):
            epoch_loss, _ = loss_loop(
                pmodel,
                train_loader,
                device,
                optimizer=self.optimizer,
                num_steps=100,
                optimizer_interval=optimizer_interval,
            )
            self.step += len(epoch_loss)

            with torch.no_grad():
                epoch_test_loss, loss_dict = loss_loop(pmodel, test_loader, device)
            loss_hist.append([np.mean(epoch_loss), np.mean(epoch_test_loss)])

            self.save_checkpoint(self.checkpoint_path)
            # if self.step % 1000 == 0:
            #     model_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.step}.pt")
            #     self.save_checkpoint(model_path)

            w_test = loss_dict["w"]
            alignment_path = os.path.join(self.checkpoint_dir, f"alignment_{self.step}.png")
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            axes.imshow(np.power(w_test[0].numpy(), 0.25).T, origin="lower")
            axes.set_title(f"Step: {self.step} | Train loss: {np.mean(epoch_loss):.2f}")
            fig.tight_layout()
            fig.savefig(alignment_path, bbox_inches="tight")
            plt.close(fig)
