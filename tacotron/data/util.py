import torch


def lengths_to_mask(lengths):
    mask = [torch.ones(x, dtype=torch.bool, device=lengths.device) for x in lengths]
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
    return mask
