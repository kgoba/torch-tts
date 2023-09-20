import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from tacotron import lengths_to_mask, mel_loss_fn, alignment_std_loss

# logger = logging.getLogger(__name__)


def plot_attention(w):
    fig, ax = plt.subplots(figsize=(4, 2))
    im = ax.imshow(w, aspect="equal", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()
    return fig


class TacotronTask(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def get_tb_logger(self):
        # Get tensorboard logger
        for logger in self.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                return logger.experiment

    def predict_forward(self, text_encoder, text_batch, xref=None, max_steps=400):
        encoded_text = [text_encoder.encode(text) for text in text_batch]
        c_lengths = torch.LongTensor([len(text) for text in encoded_text])
        c = [torch.LongTensor(text) for text in encoded_text]
        c = torch.nn.utils.rnn.pad_sequence(c, batch_first=True)
        xref_lengths = torch.LongTensor([len(x) for x in xref]) if xref != None else None

        y, y_post, s, out_dict = self.model(c, c_lengths, xref=xref, xref_lengths=xref_lengths)

        return y_post.detach().cpu()

    def train_forward(self, batch):
        _, c, c_lengths, x, x_lengths = batch

        xmask = lengths_to_mask(x_lengths).unsqueeze(2)

        y, y_post, s, out_dict = self.model(
            c, c_lengths, x, x_lengths, xref=x, xref_lengths=x_lengths
        )

        T = y.shape[1]
        x, xmask = x[:, :T, :], xmask[:, :T]

        loss_mel = mel_loss_fn(y, x, xmask, order=1) + mel_loss_fn(
            y.diff(dim=1), x.diff(dim=1), order=1
        )
        loss_mel_post = mel_loss_fn(y_post, x, xmask, order=1) + mel_loss_fn(
            y_post.diff(dim=1), x.diff(dim=1), order=1
        )

        pos_weight = torch.Tensor([0.1]).to(device=s.device)
        loss_stop = torch.nn.functional.binary_cross_entropy_with_logits(
            s, xmask.float(), pos_weight=pos_weight  # stop_weight,
        )

        loss_w = alignment_std_loss(out_dict["w"])
        loss_kl = out_dict["kl_loss"]

        loss = 0.8 * loss_mel + 0.2 * loss_mel_post + 0.1 * loss_stop
        loss += 0.0001 * loss_w
        loss += 0.0001 * loss_kl

        return (
            loss,
            {
                "total": loss.item(),
                "mel_db": 120 * (loss_mel.item()),
                "mel_post_db": 120 * (loss_mel_post.item()),
                "stop": loss_stop.item(),
                "kl": loss_kl.item(),
                "w": loss_w.item(),
            },
            {
                "w": out_dict["w"].detach().cpu(),
            },
        )

    def training_step(self, batch, batch_idx):
        loss, log_dict, img_dict = self.train_forward(batch)
        for key, value in log_dict.items():
            if isinstance(value, float):
                self.log(f"loss/{key}", value, prog_bar=False, logger=True)

        self.log(f"L", log_dict["total"], prog_bar=True, logger=False)
        self.log(f"L_mel", log_dict["mel_db"], prog_bar=True, logger=False)
        self.log(f"L_mel_p", log_dict["mel_post_db"], prog_bar=True, logger=False)
        self.log(f"L_w", log_dict["w"], prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict, img_dict = self.train_forward(batch)
        for key, value in log_dict.items():
            if isinstance(value, float):
                self.log(f"val_loss/{key}", value, prog_bar=False, logger=True)

        tb_logger = self.get_tb_logger()
        if tb_logger != None:
            tb_logger.add_figure(
                f"train/w_{batch_idx}", plot_attention(img_dict["w"][0].mT), self.global_step
            )
        return loss
