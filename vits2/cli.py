import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lightning.pytorch import LightningModule, LightningDataModule, Callback, Trainer
from lightning.pytorch.cli import LightningCLI
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

import commons
from data_utils import (
    DataConfig,
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioLoader,
)
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import (
    DurationDiscriminatorV1,
    DurationDiscriminatorV2,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
import utils

logger = logging.getLogger(__name__)


class TensorBoardEvalCallback(Callback):
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        # log mel spectrograms
        for idx_mel, mel in enumerate(pl_module.validation_mels[:5], start=1):
            pl_module.logger.experiment.add_image(
                f"val_mel_{idx_mel}",
                utils.plot_spectrogram_to_numpy(mel.to("cpu").numpy()),
                global_step=trainer.global_step,
                dataformats="HWC",
            )

        for idx_mel, mel in enumerate(pl_module.inferred_mels[:5], start=1):
            pl_module.logger.experiment.add_image(
                f"inf_mel_{idx_mel}",
                utils.plot_spectrogram_to_numpy(mel.to("cpu").numpy()),
                global_step=trainer.global_step,
                dataformats="HWC",
            )

        for idx_attn, attn in enumerate(pl_module.inferred_attn[:5], start=1):
            pl_module.logger.experiment.add_image(
                f"inf_attn_{idx_attn}",
                utils.plot_alignment_to_numpy(attn.to("cpu").numpy()),
                global_step=trainer.global_step,
                dataformats="HWC",
            )

        # log audio
        for idx_audio, audio in enumerate(pl_module.validation_y_hat[:5], start=1):
            pl_module.logger.experiment.add_audio(
                f"val_audio_{idx_audio}",
                audio.to("cpu").numpy(),
                global_step=trainer.global_step,
                sample_rate=pl_module.config.sampling_rate,
            )

        for idx_audio, audio in enumerate(pl_module.inferred_y_hat[:5], start=1):
            pl_module.logger.experiment.add_audio(
                f"inf_audio_{idx_audio}",
                audio.to("cpu").numpy(),
                global_step=trainer.global_step,
                sample_rate=pl_module.config.sampling_rate,
            )


class MyDataModule(LightningDataModule):
    def __init__(
        self,
        config: DataConfig,
        root: Path,
        datasets: list[dict],
        batch_size: int = 32,
    ):
        super().__init__()
        self.config = config
        self.root = root
        self.dataset_configs = datasets
        self.batch_size = batch_size

    def prepare_data(self):
        # download, split, etc...
        datasets = []
        for dataset_config in self.dataset_configs:
            # if dataset_config["type"] == "ljspeech":
            dataset_path = self.root / dataset_config["path"]
            datasets.append(TextAudioLoader(dataset_path, self.config))

        self.dataset = ConcatDataset(datasets)
        pass

    def setup(self, stage: str = None):
        # make assignments here (val/train/test split)
        val_size = min(100, int(len(self.dataset) * 0.05))
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [len(self.dataset) - val_size, val_size],
        )
        # self.train_sampler = DistributedBucketSampler(
        #     self.train_dataset,
        #     self.batch_size,
        #     [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        #     num_replicas=1,
        #     shuffle=True,
        # )
        self.collate_fn = TextAudioCollate()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=4,
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=True,
            # batch_sampler=self.train_sampler,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=4,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
        )


@dataclass
class ModelConfig:
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    sampling_rate: int = 24000
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024

    segment_size: int = 8192
    c_fm: float = 0.2
    c_dur: float = 1.0
    c_mel: float = 10.0
    c_kl: float = 0.2

    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4])

    n_speakers: int = 0
    gin_channels: int = 0
    use_sdp: bool = True
    use_spk_conditioned_encoder: bool = False
    use_transformer_flows: bool = True
    transformer_flow_type: str = "pre_conv"
    use_noise_scaled_mas: bool = True
    mas_noise_scale_initial: float = 0.01
    noise_scale_delta: float = 2e-6
    use_spectral_norm: bool = False
    use_mel_posterior_encoder: bool = True

    lr_gen: float = 2e-4
    lr_disc: float = 2e-4
    weight_decay: float = 1e-2


class MyTrainingModule(LightningModule):
    def __init__(self, config: ModelConfig, symbols: list[str]):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.config = config

        if config.use_mel_posterior_encoder:
            logger.info("Using mel posterior encoder for VITS2")
            posterior_channels = 80  # vits2
        else:
            logger.info("Using lin posterior encoder for VITS1")
            posterior_channels = config.filter_length // 2 + 1

        all_symbols = "".join(symbols)
        n_vocab = 1 + len(all_symbols)
        logger.info(f"All symbols [{n_vocab}]: {all_symbols}")

        self.G = SynthesizerTrn(
            n_vocab=n_vocab,
            spec_channels=posterior_channels,
            segment_size=config.segment_size // config.hop_length,
            inter_channels=config.inter_channels,
            hidden_channels=config.hidden_channels,
            filter_channels=config.filter_channels,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            kernel_size=config.kernel_size,
            p_dropout=config.p_dropout,
            resblock=config.resblock,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            upsample_rates=config.upsample_rates,
            upsample_initial_channel=config.upsample_initial_channel,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            n_speakers=config.n_speakers,
            gin_channels=config.gin_channels,
            use_sdp=config.use_sdp,
            use_spk_conditioned_encoder=config.use_spk_conditioned_encoder,
            use_transformer_flows=config.use_transformer_flows,
            transformer_flow_type=config.transformer_flow_type,
        )
        self.D = MultiPeriodDiscriminator(config.use_spectral_norm)
        # if duration_discriminator_type == "dur_disc_1":
        #     net_dur_disc = DurationDiscriminatorV1(
        #         hidden_channels,
        #         hidden_channels,
        #         3,
        #         0.1,
        #         gin_channels=gin_channels,
        #     )
        # elif duration_discriminator_type == "dur_disc_2":
        #     net_dur_disc = DurationDiscriminatorV2(
        #         hidden_channels,
        #         hidden_channels,
        #         3,
        #         0.1,
        #         gin_channels=gin_channels,
        #     )
        # self.DD = net_dur_disc

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
    
        x, x_lengths, spec, spec_lengths, y, y_lengths = batch
        # batch_size = x.shape[0]
        
        mas_noise_scale = max((
            self.config.mas_noise_scale_initial
            - self.config.noise_scale_delta * self.global_step), 0
        ) if self.config.use_noise_scaled_mas else None

        (
            y_hat,
            l_length,
            attn,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (hidden_x, logw, logw_),
        ) = self.G(x, x_lengths, spec, spec_lengths, mas_noise_scale=mas_noise_scale)

        y_slice = commons.slice_segments(
            y, ids_slice * self.config.hop_length, self.config.segment_size
        )  # slice

        if self.config.use_mel_posterior_encoder:
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec.float(),
                self.config.filter_length,
                self.config.n_mel_channels,
                self.config.sampling_rate,
                self.config.mel_fmin,
                self.config.mel_fmax,
            )
        y_mel = commons.slice_segments(
            mel, ids_slice, self.config.segment_size // self.config.hop_length
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.config.filter_length,
            self.config.n_mel_channels,
            self.config.sampling_rate,
            self.config.hop_length,
            self.config.win_length,
            self.config.mel_fmin,
            self.config.mel_fmax,
        )

        ##########################
        # Optimize Discriminator #
        ##########################
        y_d_hat_r, y_d_hat_g, _, _ = self.D(y_slice, y_hat.detach())
        losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc = torch.mean(losses_disc_r) + torch.mean(losses_disc_g)

        d_opt.zero_grad()
        self.manual_backward(loss_disc)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        _, y_d_hat_g, fmap_r, fmap_g = self.D(y_slice, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel)
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

        loss_fm = feature_loss(fmap_r, fmap_g)
        losses_gen = generator_loss(y_d_hat_g)
        loss_gen = torch.mean(losses_gen)
        loss_gen_all = (
            loss_gen
            + (loss_fm * self.config.c_fm)
            + (loss_dur * self.config.c_dur)
            + (loss_kl * self.config.c_kl)
            + (loss_mel * self.config.c_mel)
        ) / (
            1
            + self.config.c_fm
            + self.config.c_dur
            + self.config.c_kl
            + self.config.c_mel
        )

        g_opt.zero_grad()
        self.manual_backward(loss_gen_all)
        g_opt.step()

        if x.device.type == "mps":
            # NB: for memory efficiency use mimalloc (see https://github.com/pytorch/pytorch/issues/111517)
            torch.mps.empty_cache()
            gc.collect()

        self.log_dict(
            {
                "L/fm": loss_fm,
                "L/kl": loss_kl,
                "L/dur": loss_dur,
            },
            prog_bar=False,
        )

        self.log_dict(
            {
                "L/d": loss_disc,
                "L/g": loss_gen,
                "L/g_all": loss_gen_all,
                "L/mel": loss_mel,
            },
            prog_bar=True,
        )

    def on_validation_start(self) -> None:
        self.validation_mels = []
        self.validation_y_hat = []
        self.inferred_mels = []
        self.inferred_y_hat = []
        self.inferred_attn = []

    def validation_step(self, batch, batch_idx):
        x, x_lengths, spec, spec_lengths, y, y_lengths = batch

        (
            y_hat,
            l_length,
            attn,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (hidden_x, logw, logw_),
        ) = self.G(x, x_lengths, spec, spec_lengths)

        if self.config.use_mel_posterior_encoder:
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec,
                self.config.filter_length,
                self.config.n_mel_channels,
                self.config.sampling_rate,
                self.config.mel_fmin,
                self.config.mel_fmax,
            )

        y_mel = commons.slice_segments(
            mel, ids_slice, self.config.segment_size // self.config.hop_length
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.config.filter_length,
            self.config.n_mel_channels,
            self.config.sampling_rate,
            self.config.hop_length,
            self.config.win_length,
            self.config.mel_fmin,
            self.config.mel_fmax,
        )

        for y_hat_i in y_hat:
            self.validation_y_hat.append(y_hat_i)
        for mel_i in y_hat_mel:
            self.validation_mels.append(mel_i)

        loss_dur = torch.sum(l_length.float())
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel)

        self.log_dict({"VL/dur": loss_dur, "VL/mel": loss_mel})

        # Inference
        if batch_idx == 0:
            y_hat, attn, mask, *_ = self.G.infer(x, x_lengths, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * self.config.hop_length

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                self.config.filter_length,
                self.config.n_mel_channels,
                self.config.sampling_rate,
                self.config.hop_length,
                self.config.win_length,
                self.config.mel_fmin,
                self.config.mel_fmax,
            )
            for mel_i in y_hat_mel:
                self.inferred_mels.append(mel_i)
            for y_hat_i, len_i in zip(list(y_hat), list(y_hat_lengths)):
                self.inferred_y_hat.append(y_hat_i[:, :len_i])
            for attn_i in attn:
                self.inferred_attn.append(attn_i[0])

    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(
            self.G.parameters(),
            lr=self.config.lr_gen,
            weight_decay=self.config.weight_decay,
        )
        d_opt = torch.optim.AdamW(
            self.D.parameters(),
            lr=self.config.lr_disc,
            weight_decay=self.config.weight_decay,
        )
        return g_opt, d_opt


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.config.symbols", "model.symbols", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.config.hop_length", "model.config.hop_length", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.config.win_length", "model.config.win_length", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.config.sampling_rate",
            "model.config.sampling_rate",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.config.n_mel_channels",
            "model.config.n_mel_channels",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.config.use_mel_posterior_encoder",
            "model.config.use_mel_posterior_encoder",
            apply_on="instantiate",
        )


def cli_main():
    try:
        torch.set_float32_matmul_precision("medium")
        logger.info("TensorCore activated")
    except Exception as e:
        logger.info("TensorCore not activated")
        pass
    cli = MyLightningCLI(MyTrainingModule, MyDataModule)  # , BoringDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
