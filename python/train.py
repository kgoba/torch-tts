import torch
import numpy as np
import yaml, os, sys, logging
from tqdm.auto import tqdm

from audio import AudioFrontend, AudioFrontendConfig
from dataset import TranscribedAudioDataset, TacotronDataset, TextEncoder, collate_fn_tacotron
from tacotron import build_tacotron

logger = logging.getLogger(__name__)


def find_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def check_dataset_stats(dataset, sample_size=500):
    print(f"Dataset size: {len(dataset)}")
    sample, _ = torch.utils.data.random_split(
        dataset,
        [sample_size, len(dataset) - sample_size],
        generator=torch.Generator().manual_seed(142),
    )
    utt_len = []
    audio_len = []
    audio_pwr = []
    for i in tqdm(range(len(sample))):
        utt_id, transcript, audio, sr = dataset[i]
        utt_len.append(len(transcript))
        audio_len.append(len(audio) / sr)
        audio_pwr.append(10 * np.log10(np.mean(audio.numpy() ** 2)))

    print(
        f"Utterance length: {np.median(utt_len):.1f} (median), {np.quantile(utt_len, 0.05):.1f}..{np.quantile(utt_len, 0.95):.1f} (5%..95%) characters"
    )
    print(
        f"Audio length:     {np.median(audio_len):.1f} (median), {np.quantile(audio_len, 0.05):.1f}..{np.quantile(audio_len, 0.95):.1f} (5%..95%) s"
    )
    print(
        f"Audio RMS power:  {np.median(audio_pwr):.1f} (median), {np.quantile(audio_pwr, 0.05):.1f}..{np.quantile(audio_pwr, 0.95):.1f} (5%..95%) dBFS"
    )
    print(
        f"Total audio length: {len(dataset) * np.mean(audio_len) / 3600:.1f} h (estimated)"
    )

# def loss_fn(outputs, labels, mask):
#     loss = (outputs - labels).abs() * mask # nn.functional.l1_loss(outputs, labels, reduction='none') * mask
#     loss = loss.sum(dim=(1,2)) / (mask.sum(dim=(1,2)) * outputs.size(-1))
#     return loss.mean() # + loss.max()) / 2

# def loss_loop(model, loss_fn, loader, optimizer=None):
#     if optimizer is not None: model.train()
#     else: model.eval()
#     loss_hist = []
#     for batch_idx, batch in enumerate(loader):
#         inputs, labels, mask = [x.to(device) for x in batch]
#         if optimizer is not None:
#             optimizer.zero_grad()
#         outputs = model(inputs) # [0]
#         loss = loss_fn(outputs, labels, mask)
#         if optimizer is not None:
#             loss.backward()
#             optimizer.step()
#         loss_hist.append(loss.item() * 15)
#     return loss_hist


def main(args):
    dataset_path = args.dataset
    config_path = args.config

    test_size = 200

    config = yaml.safe_load(open(config_path))
    random_seed = config["seed"] or 42

    # dataset = build_dataset(dataset_path, config)
    audio_dataset = TranscribedAudioDataset(
        os.path.join(dataset_path, "transcripts.txt"),
        dataset_path,
        filename_fn=lambda x: x + ".wav",
    )
    audio_frontend = AudioFrontend(AudioFrontendConfig().from_json(config["audio"]))
    text_encoder = TextEncoder(config["text"]["alphabet"])
    dataset = TacotronDataset(audio_dataset, audio_frontend, text_encoder)

    check_dataset_stats(audio_dataset)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    print(f"Dataset size: {len(train_dataset)} train + {len(test_dataset)} test")

    device = find_available_device()
    print(f"Using device {device}")
    print(f"Number of CPUs: {os.cpu_count()}")

    device = 'mps'

    # train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=100, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn_tacotron, batch_size=16, pin_memory=True)

    model = build_tacotron(config)
    model.to(device)

    r = 2
    for batch in test_loader:
        input, imask, x, xmask = batch

        T = (x.shape[1] // r) * r
        x = x[:, :T, :]
        xmask = xmask[:, :T, :]

        print(torch.min(x), torch.mean(x), torch.median(x), torch.max(x))

        x = x.to(device)
        xmask = xmask.to(device)

        y, w = model(input.to(device), imask.to(device), x)
        # print(x.shape, y.shape, y.dtype)
        print(x.device, y.device, xmask.device)

        loss = torch.mean(torch.nn.functional.l1_loss(x, y, reduction='none'))  #* xmask)
        print(loss.shape, loss)
        break

    # model = MelToSTFTModel(80, 241)
    # # model = Autoencoder(STFTEncoder(241, 16), MelToSTFTModel(16, 241))
    # print(f"Model structure: {model}")

    # model2 = model if device == 'cpu' else nn.DataParallel(model)
    # optimizer = torch.optim.AdamW(model2.parameters(), lr=0.001, amsgrad=True, weight_decay=0.01)

    # model2.to(device)

    # pbar = tqdm(range(600))
    # loss_hist = []
    # for epoch_idx in pbar:
    #     epoch_loss = loss_loop(model2, loss_fn, train_loader, optimizer)
    #     epoch_test_loss = loss_loop(model2, loss_fn, test_loader, None)
    #     pbar.set_postfix_str(f"Loss: {np.mean(epoch_loss):.2f} dB, Test: {np.mean(epoch_test_loss):.2f} dB")
    #     loss_hist.append([np.mean(epoch_loss), np.mean(epoch_test_loss)])

    # model2.to('cpu')
    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save('/kaggle/working/model_scripted.pt') # Save

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset path")
    parser.add_argument("config", help="Configuration file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    rc = main(args)
    sys.exit(rc)
