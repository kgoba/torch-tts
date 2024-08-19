# VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design
### Jungil Kong, Jihoon Park, Beomjeong Kim, Jeongmin Kim, Dohee Kong, Sangjin Kim
Unofficial implementation of the [VITS2 paper](https://arxiv.org/abs/2307.16430), sequel to [VITS paper](https://arxiv.org/abs/2106.06103). (thanks to the authors for their work!)

![Alt text](resources/image.png)

## Credits
- We will build this repo based on the [VITS repo](https://github.com/jaywalnut310/vits). The goal is to make this model easier to transfer learning from VITS pretrained model!
- (08-17-2023) - The authors were really kind to guide me through the paper and answer my questions. I am open to discuss any changes or answer questions regarding the implementation. Please feel free to open an issue or contact me directly.
- [VITS2 code](https://github.com/p0p4k/vits2_pytorch)

## Prerequisites
1. Python >= 3.10
2. Tested on Pytorch version 1.13.1 with Google Colab and LambdaLabs cloud.
3. Clone this repository
4. Install python requirements. Please refer to [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
5. Download datasets
    1. Download and extract the LJ Speech dataset.
    1. For mult-speaker setting, download and extract the VCTK dataset., and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
6. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.

```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace
```

## TODOs, features and notes

#### Duration predictor (fig 1a)
- [x] Added LSTM discriminator to duration predictor.
- [x] Added adversarial loss to duration predictor. ("use_duration_discriminator" flag in config file; default is "True")
- [x] Monotonic Alignment Search with Gaussian Noise added; might need expert verification (Section 2.2)
- [x] Added "use_noise_scaled_mas" flag in config file. Choose from True or False; updates noise while training based on number of steps and never goes below 0.0
- [x] Update models.py/train.py/train_ms.py
- [x] Update config files (vits2_vctk_base.json; vits2_ljs_base.json)
- [x] Update losses in train.py and train_ms.py
#### Transformer block in the normalizing flow (fig 1b)
- [x] Added transformer block to the normalizing flow. There are three types of transformer blocks: pre-convolution (my implementation), FFT (from [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc/commit/fc8336fffd40c39bdb225c1b041ab4dd15fac4e9) repo) and mono-layer.
- [x] Added "transformer_flow_type" flag in config file. Choose from "pre_conv", "fft", "mono_layer_inter_residual", "mono_layer_post_residual".
- [x] Added layers and blocks in models.py
(ResidualCouplingTransformersLayer,
ResidualCouplingTransformersBlock,
FFTransformerCouplingLayer,
MonoTransformerFlowLayer)
- [x] Add in config file (vits2_ljs_base.json; can be turned on using "use_transformer_flows" flag)
#### Speaker-conditioned text encoder (fig 1c)
- [x] Added speaker embedding to the text encoder in models.py (TextEncoder; backward compatible with VITS)
- [x] Add in config file (vits2_ljs_base.json; can be turned on using "use_spk_conditioned_encoder" flag)
#### Mel spectrogram posterior encoder (Section 3)
- [x] Added mel spectrogram posterior encoder in train.py
- [x] Addded new config file (vits2_ljs_base.json; can be turned on using "use_mel_posterior_encoder" flag)
- [x] Updated 'data_utils.py' to use the "use_mel_posterior_encoder" flag for vits2
#### Training scripts
- [x] Added vits2 flags to train.py (single-speaer model)
- [x] Added vits2 flags to train_ms.py (multi-speaker model)
#### ONNX export
- [x] Add ONNX export support.
#### Gradio Demo
- [x] Add Gradio demo support.

