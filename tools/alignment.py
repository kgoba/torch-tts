import random
from matplotlib import pyplot as plt

import torch


def compute_alignment_probabilities(P: torch.Tensor, num_frames: int):
    """
    Compute the alignment probabilities between a phoneme string and acoustic frames.

    Args:
        P (torch.Tensor): Input tensor of phoneme duration probabilities, shape (num_phonemes, max_duration).
        num_frames (int): Number of acoustic frames.

    Returns:
        torch.Tensor: Alignment probability matrix A, shape (num_frames, num_phonemes).
    """
    num_phonemes = P.shape[0]
    max_duration = P.shape[1] - 1

    # Compute the cumulative duration probabilities
    Q = torch.zeros(num_phonemes, num_frames)
    Q[0, : max_duration + 1] = P[0, :]
    for i in range(1, num_phonemes):
        for j in range(num_frames):
            for m in range(max(0, j - max_duration), j + 1):
                Q[i, j] += Q[i - 1, m] * P[i, j - m]
    print(Q.sum(dim=1))

    Pcum = torch.cumsum(P.flip(1), dim=1).flip(1)

    # Compute the alignment probabilities
    A = torch.zeros(num_phonemes, num_frames + 1)
    A[0, : max_duration + 1] = Pcum[0, :]
    for i in range(1, num_phonemes):
        for j in range(num_frames + 1):
            for m in range(max(0, j - max_duration), j):
                A[i, j] += Q[i - 1, m] * Pcum[i, j - m]
    A = A[:, 1:]
    print(A.sum(dim=0))

    return Q, A


def compute_alignment_probabilities_by_convolution(P: torch.Tensor, num_frames: int):
    num_phonemes = P.shape[0]
    max_duration = P.shape[1] - 1

    # Compute the cumulative duration probabilities
    Q = torch.zeros(num_phonemes, num_frames)
    Q[0, : max_duration + 1] = P[0, :]
    for i in range(1, num_phonemes):
        Q[i] = torch.nn.functional.conv1d(
            Q[i - 1].unsqueeze(0).unsqueeze(1),
            P[i].unsqueeze(0).unsqueeze(1).flip(2),
            padding=max_duration,
        )[0, 0, :num_frames]
    print(Q.sum(dim=1))

    Pcum = torch.cumsum(P.flip(1), dim=1).flip(1)

    # Compute the alignment probabilities
    Qrow = torch.zeros(1, num_frames)
    Qrow[:, 0] = 1
    A = torch.nn.functional.conv1d(
        torch.cat([Qrow, Q[:-1]], dim=0).unsqueeze(0),
        Pcum[:, 1:].unsqueeze(1).flip(2),
        padding=max_duration,
        groups=num_phonemes,
    )[0, :, 1 : num_frames + 1]
    print(A.sum(dim=0))

    return Q, A


def plot_alignment_probabilities(Q, A):
    """
    Plot the alignment probabilities.

    Args:
        A (torch.Tensor): Alignment probability matrix A, shape (num_frames, num_phonemes).
    """
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(
        Q,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.xlabel("Frames")
    plt.ylabel("Phonemes")
    plt.title("Duration Probabilities")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(
        A,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.xlabel("Frames")
    plt.ylabel("Phonemes")
    plt.title("Alignment Probabilities")
    plt.colorbar()


def generate_random_phoneme_duration_probabilities(
    num_phonemes: int, max_duration: int
):
    """
    Generate a random phoneme duration probability tensor.

    Args:
        num_phonemes (int): Number of phonemes.
        max_duration (int): Maximum phoneme duration.

    Returns:
        torch.Tensor: Random phoneme duration probability tensor, shape (num_phonemes, max_duration).
    """
    x = torch.randn(num_phonemes, max_duration + 1)
    for i in range(num_phonemes):
        x[i, random.randint(0, max_duration)] = 10
    # x = 5 * torch.randn(num_phonemes, max_duration + 1)
    P = x.softmax(dim=1)
    # P = x / x.sum(dim=1, keepdim=True)
    return P


def generate_test_case1():
    num_phonemes = 5
    max_duration = 3
    P = torch.zeros(num_phonemes, max_duration + 1)
    P[0, 1] = 1
    P[1, 1] = 0.8
    P[1, 2] = 0.2
    P[2, 1] = 0.8
    P[2, 2] = 0.2
    P[3, 3] = 1
    P[4, 1] = 0.9
    P[4, 2] = 0.1
    return P


def generate_test_case2():
    num_phonemes = 10
    max_duration = 1
    P = torch.zeros(num_phonemes, max_duration + 1)
    P[:, 1] = 1
    P[0, 1] = 0.5
    P[0, 0] = 0.5
    return P


# Generate a random phoneme duration probability tensor
P = generate_random_phoneme_duration_probabilities(num_phonemes=10, max_duration=4)
# P = generate_test_case2()
# plot_alignment_probabilities(P)
# Test the alignment probability computation
num_frames = P.size(0) * (P.size(1) - 1)
Qr, Ar = compute_alignment_probabilities(P, num_frames)
Q, A = compute_alignment_probabilities_by_convolution(P, num_frames)
assert Qr.shape == Q.shape
assert Ar.shape == A.shape
err_Q = torch.norm(Qr - Q)
err_A = torch.norm(Ar - A)
if err_Q < 1e-6 and err_A < 1e-6:
    print("Test passed.")
else:
    plot_alignment_probabilities(Qr, Ar)
    plot_alignment_probabilities(Q, A)
    plt.show()
