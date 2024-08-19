import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lightning.pytorch import LightningModule, LightningDataModule
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

logger = logging.getLogger(__name__)


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
        val_size = min(200, int(len(self.dataset) * 0.05))
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
            num_workers=2,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            # batch_sampler=self.train_sampler,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=2,
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
    c_mel: float = 45.0
    c_kl: float = 1.0

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
    use_transformer_flows: bool = False
    transformer_flow_type: str = "mono_layer_post_residual"
    use_noise_scaled_mas: bool = False
    mas_noise_scale_initial: float = 0.01
    noise_scale_delta: float = 2e-6
    use_spectral_norm: bool = False
    use_mel_posterior_encoder: bool = False

    lr_gen: float = 1e-5
    lr_disc: float = 1e-5
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
            use_noise_scaled_mas=config.use_noise_scaled_mas,
            mas_noise_scale_initial=config.mas_noise_scale_initial,
            noise_scale_delta=config.noise_scale_delta,
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
        losses_disc_r, losses_disc_g = discriminator_loss(
            y_d_hat_r, y_d_hat_g
        )
        loss_disc = torch.mean(losses_disc_r) + torch.mean(losses_disc_g)

        d_opt.zero_grad()
        self.manual_backward(loss_disc)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        with torch.no_grad():
            _, y_d_hat_g, fmap_r, fmap_g = self.D(y_slice, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel)
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

        loss_fm = feature_loss(fmap_r, fmap_g)
        losses_gen = generator_loss(y_d_hat_g)
        loss_gen = torch.mean(losses_gen)
        loss_gen_all = (
            loss_gen
            + loss_fm
            + loss_dur
            + (loss_kl * self.config.c_kl)
            + (loss_mel * self.config.c_mel)
        ) / (3 + self.config.c_kl + self.config.c_mel)

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

        loss_dur = torch.sum(l_length.float())
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel)

        self.log_dict({"VL/dur": loss_dur, "VL/mel": loss_mel})

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
    cli = MyLightningCLI(MyTrainingModule, MyDataModule)  # , BoringDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
