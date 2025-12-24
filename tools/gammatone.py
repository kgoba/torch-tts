# Copyright 2014 Jason Heeris, jason.heeris@gmail.com
#
# This file is part of the gammatone toolkit, and is licensed under the 3-clause
# BSD license: https://github.com/detly/gammatone/blob/master/COPYING
"""
This module contains functions for constructing sets of equivalent rectangular
bandwidth gammatone filters.
"""
from __future__ import division
from collections import namedtuple

import numpy as np
import scipy as sp
from scipy import signal as sgn

DEFAULT_FILTER_NUM = 100
DEFAULT_LOW_FREQ = 100
DEFAULT_HIGH_FREQ = 44100 / 4


def erb_point(low_freq, high_freq, fraction):
    """
    Calculates a single point on an ERB scale between ``low_freq`` and
    ``high_freq``, determined by ``fraction``. When ``fraction`` is ``1``,
    ``low_freq`` will be returned. When ``fraction`` is ``0``, ``high_freq``
    will be returned.

    ``fraction`` can actually be outside the range ``[0, 1]``, which in general
    isn't very meaningful, but might be useful when ``fraction`` is rounded a
    little above or below ``[0, 1]`` (eg. for plot axis labels).
    """
    # Change the following three parameters if you wish to use a different ERB
    # scale. Must change in MakeERBCoeffs too.
    # TODO: Factor these parameters out
    ear_q = 9.26449  # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1

    # All of the following expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear Filter
    # Bank." See pages 33-34.
    erb_point = -ear_q * min_bw + np.exp(
        fraction * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw))
    ) * (high_freq + ear_q * min_bw)

    return erb_point


def erb_space(low_freq=DEFAULT_LOW_FREQ, high_freq=DEFAULT_HIGH_FREQ, num=DEFAULT_FILTER_NUM):
    """
    This function computes an array of ``num`` frequencies uniformly spaced
    between ``high_freq`` and ``low_freq`` on an ERB scale.

    For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
    "Suggested formulae for calculating auditory-filter bandwidths and
    excitation patterns," J. Acoust. Soc. Am. 74, 750-753.
    """
    return erb_point(low_freq, high_freq, np.arange(1, num + 1) / num)


def centre_freqs(fs, num_freqs, cutoff):
    """
    Calculates an array of centre frequencies (for :func:`make_erb_filters`)
    from a sampling frequency, lower cutoff frequency and the desired number of
    filters.

    :param fs: sampling rate
    :param num_freqs: number of centre frequencies to calculate
    :type num_freqs: int
    :param cutoff: lower cutoff frequency
    :return: same as :func:`erb_space`
    """
    return np.flipud(erb_space(cutoff, fs / 2, num_freqs))
    # return np.exp(np.log(cutoff) + np.log(fs / 2 / cutoff) * np.linspace(0, 1, num_freqs))


def make_erb_filters(fs, centre_freqs, width=1.0):
    """
    This function computes the filter coefficients for a bank of
    Gammatone filters. These filters were defined by Patterson and Holdworth for
    simulating the cochlea.

    The result is returned as a :class:`ERBCoeffArray`. Each row of the
    filter arrays contains the coefficients for four second order filters. The
    transfer function for these four filters share the same denominator (poles)
    but have different numerators (zeros). All of these coefficients are
    assembled into one vector that the ERBFilterBank can take apart to implement
    the filter.

    The filter bank contains "numChannels" channels that extend from
    half the sampling rate (fs) to "lowFreq". Alternatively, if the numChannels
    input argument is a vector, then the values of this vector are taken to be
    the center frequency of each desired filter. (The lowFreq argument is
    ignored in this case.)

    Note this implementation fixes a problem in the original code by
    computing four separate second order filters. This avoids a big problem with
    round off errors in cases of very small cfs (100Hz) and large sample rates
    (44kHz). The problem is caused by roundoff error when a number of poles are
    combined, all very close to the unit circle. Small errors in the eigth order
    coefficient, are multiplied when the eigth root is taken to give the pole
    location. These small errors lead to poles outside the unit circle and
    instability. Thanks to Julius Smith for leading me to the proper
    explanation.

    Execute the following code to evaluate the frequency response of a 10
    channel filterbank::

        fcoefs = MakeERBFilters(16000,10,100);
        y = ERBFilterBank([1 zeros(1,511)], fcoefs);
        resp = 20*log10(abs(fft(y')));
        freqScale = (0:511)/512*16000;
        semilogx(freqScale(1:255),resp(1:255,:));
        axis([100 16000 -60 0])
        xlabel('Frequency (Hz)'); ylabel('Filter Response (dB)');

    | Rewritten by Malcolm Slaney@Interval.  June 11, 1998.
    | (c) 1998 Interval Research Corporation
    |
    | (c) 2012 Jason Heeris (Python implementation)
    """
    T = 1 / fs
    # Change the followFreqing three parameters if you wish to use a different
    # ERB scale. Must change in ERBSpace too.
    # TODO: factor these out
    ear_q = 9.26449  # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1

    erb = width * ((centre_freqs / ear_q) ** order + min_bw**order) ** (1 / order)
    # erb = width * centre_freqs / 5
    B = 1.019 * 2 * np.pi * erb

    arg = 2 * centre_freqs * np.pi * T
    vec = np.exp(2j * arg)

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * np.cos(arg) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)

    rt_pos = np.sqrt(3 + 2**1.5)
    rt_neg = np.sqrt(3 - 2**1.5)

    common = -T * np.exp(-(B * T))

    # TODO: This could be simplified to a matrix calculation involving the
    # constant first term and the alternating rt_pos/rt_neg and +/-1 second
    # terms
    k11 = np.cos(arg) + rt_pos * np.sin(arg)
    k12 = np.cos(arg) - rt_pos * np.sin(arg)
    k13 = np.cos(arg) + rt_neg * np.sin(arg)
    k14 = np.cos(arg) - rt_neg * np.sin(arg)

    A11 = common * k11
    A12 = common * k12
    A13 = common * k13
    A14 = common * k14

    gain_arg = np.exp(1j * arg - B * T)

    gain = np.abs(
        (vec - gain_arg * k11)
        * (vec - gain_arg * k12)
        * (vec - gain_arg * k13)
        * (vec - gain_arg * k14)
        * (T * np.exp(B * T) / (-1 / np.exp(B * T) + 1 + vec * (1 - np.exp(B * T)))) ** 4
    )

    allfilts = np.ones_like(centre_freqs)

    # return (
    #     np.column_stack([A0 * allfilts, A11, A12, A13, A14, A2 * allfilts]),
    #     np.column_stack([B0 * allfilts, B1, B2]),
    #     gain,
    # )
    fcoefs = np.column_stack(
        [A0 * allfilts, A11, A12, A13, A14, A2 * allfilts, B0 * allfilts, B1, B2, gain]
    )
    return fcoefs


def erb_filterbank(wave, coefs):
    """
    :param wave: input data (one dimensional sequence)
    :param coefs: gammatone filter coefficients

    Process an input waveform with a gammatone filter bank. This function takes
    a single sound vector, and returns an array of filter outputs, one channel
    per row.

    The fcoefs parameter, which completely specifies the Gammatone filterbank,
    should be designed with the :func:`make_erb_filters` function.

    | Malcolm Slaney @ Interval, June 11, 1998.
    | (c) 1998 Interval Research Corporation
    | Thanks to Alain de Cheveigne' for his suggestions and improvements.
    |
    | (c) 2013 Jason Heeris (Python implementation)
    """
    output = np.zeros((coefs[:, 9].shape[0], wave.shape[0]))

    gain = coefs[:, 9]
    # A0, A11, A2
    As1 = coefs[:, (0, 1, 5)]
    # A0, A12, A2
    As2 = coefs[:, (0, 2, 5)]
    # A0, A13, A2
    As3 = coefs[:, (0, 3, 5)]
    # A0, A14, A2
    As4 = coefs[:, (0, 4, 5)]
    # B0, B1, B2
    Bs = coefs[:, 6:9]

    # Loop over channels
    for idx in range(0, coefs.shape[0]):
        # These seem to be reversed (in the sense of A/B order), but that's what
        # the original code did...
        # Replacing these with polynomial multiplications reduces both accuracy
        # and speed.
        y1 = sgn.lfilter(As1[idx], Bs[idx], wave)
        y2 = sgn.lfilter(As2[idx], Bs[idx], y1)
        y3 = sgn.lfilter(As3[idx], Bs[idx], y2)
        y4 = sgn.lfilter(As4[idx], Bs[idx], y3)
        output[idx, :] = y4 / gain[idx]

    return output


def gtgram_xe(wave, fs, channels, f_min):
    """Calculate the intermediate ERB filterbank processed matrix"""
    cfs = centre_freqs(fs, channels, f_min)
    fcoefs = np.flipud(make_erb_filters(fs, cfs, width=0.5))
    xf = erb_filterbank(wave, fcoefs)
    xe = np.power(xf, 2)
    return xe


def round_half_away_from_zero(num):
    """ Implement the round-half-away-from-zero rule, where fractional parts of
    0.5 result in rounding up to the nearest positive integer for positive
    numbers, and down to the nearest negative number for negative integers.
    """
    return np.sign(num) * np.floor(np.abs(num) + 0.5)

def gtgram_strides(fs, window_time, hop_time, filterbank_cols):
    """
    Calculates the window size for a gammatonegram.

    @return a tuple of (window_size, hop_samples, output_columns)
    """
    nwin        = int(round_half_away_from_zero(window_time * fs))
    hop_samples = int(round_half_away_from_zero(hop_time * fs))
    columns     = (1
                    + int(
                        np.floor(
                            (filterbank_cols - nwin)
                            / hop_samples
                        )
                    )
                  )

    return (nwin, hop_samples, columns)

def gtgram(
    wave,
    fs,
    window_time, hop_time,
    channels,
    f_min):
    """
    Calculate a spectrogram-like time frequency magnitude array based on
    gammatone subband filters. The waveform ``wave`` (at sample rate ``fs``) is
    passed through an multi-channel gammatone auditory model filterbank, with
    lowest frequency ``f_min`` and highest frequency ``f_max``. The outputs of
    each band then have their energy integrated over windows of ``window_time``
    seconds, advancing by ``hop_time`` secs for successive columns. These
    magnitudes are returned as a nonnegative real matrix with ``channels`` rows.

    | 2009-02-23 Dan Ellis dpwe@ee.columbia.edu
    |
    | (c) 2013 Jason Heeris (Python implementation)
    """
    xe = gtgram_xe(wave, fs, channels, f_min)

    nwin, hop_samples, ncols = gtgram_strides(
        fs,
        window_time,
        hop_time,
        xe.shape[1]
    )
    win = np.hanning(nwin)

    y = np.zeros((channels, ncols))

    for cnum in range(ncols):
        segment = xe[:, cnum * hop_samples + np.arange(nwin)] * win
        y[:, cnum] = np.sqrt(segment.mean(1))

    return y


def plot_filterbank(fs, channels, f_min):
    from matplotlib import pyplot as plt

    fs = 22050
    channels = 80
    f_min = 50
    cfs = centre_freqs(fs, channels, f_min)
    fcoefs = make_erb_filters(fs, cfs, width=0.2)
    print(cfs)
    f = np.logspace(np.log10(f_min), np.log10(fs / 2), 500)
    w = 2 * np.pi * f / fs
    for i in range(channels):
        z = np.exp(1j * w)
        H_p1 = np.polyval((fcoefs[i, (0, 1, 5)]), z)
        H_p2 = np.polyval((fcoefs[i, (0, 2, 5)]), z)
        H_p3 = np.polyval((fcoefs[i, (0, 3, 5)]), z)
        H_p4 = np.polyval((fcoefs[i, (0, 4, 5)]), z)
        H_q = np.polyval(fcoefs[i, 6:9], z)
        H = (H_p1 / H_q) * (H_p2 / H_q) * (H_p3 / H_q) * (H_p4 / H_q)
        H /= fcoefs[i, 9]
        A = 20 * np.log10(np.abs(H))
        plt.semilogx(f, A)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # plot_filterbank(22050, 80, 50)
    import torchaudio
    import torch
    import sys

    wave, sr = torchaudio.load(sys.argv[1])
    wave = torchaudio.functional.resample(wave, sr, 16000)
    sr = 16000

    wave = wave.mean(dim=0)
    # g = gtgram_xe(wave.numpy(), sr, 80, 50)
    g = gtgram(wave.numpy(), sr, 0.036, 0.012, 80, 60)
    print(g.shape)

    n_fft = 576
    s2 = torchaudio.functional.spectrogram(wave, 0, torch.hann_window(n_fft), n_fft, 192, n_fft, power=2, normalized=True)
    s1 = torchaudio.functional.spectrogram(wave, 0, torch.hann_window(n_fft), n_fft, 192, n_fft, power=1, normalized=True)
    m = torchaudio.functional.melscale_fbanks(n_fft//2 + 1, 60, sr/2, 80, sr, mel_scale="slaney")
    m1 = m.mT.matmul(s1)
    m2 = m.mT.matmul(s2)


    from matplotlib import pyplot as plt

    plt.subplot(311)
    plt.imshow(10 * np.log10(g), origin="upper")
    plt.subplot(312)
    plt.imshow(20 * np.log10(m1), origin="lower")
    plt.subplot(313)
    plt.imshow(10 * np.log10(m2), origin="lower")
    plt.show()
