import torch
import torchaudio
from dataclasses import dataclass

import sys
import matplotlib
import matplotlib.pyplot as plt

from torchaudio.models import wav2vec2_model


print(torch.__version__)
print(torchaudio.__version__)


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"

print(device)

try:
    from torchaudio.functional import forced_align
except ImportError:
    print(
        "Failed to import the forced alignment API. "
        "Please install torchaudio nightly builds. "
        "Please refer to https://pytorch.org/get-started/locally "
        "for instructions to install a nightly build."
    )
    raise

import torchaudio.functional as F


@dataclass
class Frame:
    # This is the index of each token in the transcript,
    # i.e. the current frame aligns to the N-th character from the transcript.
    token_index: int
    time_index: int
    score: float


def compute_alignments(transcript, dictionary, emission):
    frames = []
    tokens = [dictionary[c] for c in transcript.replace(" ", "")]

    targets = torch.tensor(tokens, dtype=torch.int32)
    input_lengths = torch.tensor(emission.shape[0])
    target_lengths = torch.tensor(targets.shape[0])

    # This is the key step, where we call the forced alignment API functional.forced_align to compute alignments.
    frame_alignment, frame_scores = F.forced_align(
        emission, targets, input_lengths, target_lengths, 0
    )

    assert len(frame_alignment) == input_lengths.item()
    assert len(targets) == target_lengths.item()

    token_index = -1
    prev_hyp = 0
    for i in range(len(frame_alignment)):
        if frame_alignment[i].item() == 0:
            prev_hyp = 0
            continue

        if frame_alignment[i].item() != prev_hyp:
            token_index += 1
        frames.append(Frame(token_index, i, frame_scores[i].exp().item()))
        prev_hyp = frame_alignment[i].item()
    return frames, frame_alignment, frame_scores


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(frames, transcript):
    transcript_nospace = transcript.replace(" ", "")
    i1, i2 = 0, 0
    segments = []
    while i1 < len(frames):
        while i2 < len(frames) and frames[i1].token_index == frames[i2].token_index:
            i2 += 1
        score = sum(frames[k].score for k in range(i1, i2)) / (i2 - i1)
        tokens = [
            dictionary[c] if c in dictionary else dictionary["@"]
            for c in transcript.replace(" ", "")
        ]

        segments.append(
            Segment(
                transcript_nospace[frames[i1].token_index],
                frames[i1].time_index,
                frames[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


# Obtain word alignments from token alignments
def merge_words(transcript, segments, separator=" "):
    words = []
    i1, i2, i3 = 0, 0, 0
    while i3 < len(transcript):
        if i3 == len(transcript) - 1 or transcript[i3] == separator:
            if i1 != i2:
                if i3 == len(transcript) - 1:
                    i2 += 1
                if separator == "|":
                    # s is the number of separators (counted as a valid modeling unit) we've seen
                    s = len(words)
                else:
                    s = 0
                segs = segments[i1 + s : i2 + s]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(
                    seg.length for seg in segs
                )
                words.append(
                    Segment(word, segments[i1 + s].start, segments[i2 + s - 1].end, score)
                )
            i1 = i2
        else:
            i2 += 1
        i3 += 1
    return words


def plot_alignments(segments, word_segments, waveform, input_lengths, scale=10):
    fig, ax2 = plt.subplots() # figsize=(64, 12))
    # plt.rcParams.update({"font.size": 30})

    # The original waveform
    ratio = waveform.size(0) / input_lengths
    ax2.plot(waveform)
    ax2.set_ylim(-1.0 * scale, 1.0 * scale)
    ax2.set_xlim(0, waveform.size(-1))

    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, alpha=0.1, color="red")
        ax2.annotate(f"{word.score:.2f}", (x0, 0.8 * scale))

    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, 0.9 * scale))

    xticks = ax2.get_xticks()
    plt.xticks(xticks, xticks / sample_rate) # , fontsize=50)
    ax2.set_xlabel("time [second]") # , fontsize=40)
    ax2.set_yticks([])


def compute_and_plot_alignments(transcript, dictionary, emission, waveform):
    frames, frame_alignment, _ = compute_alignments(transcript, dictionary, emission)
    segments = merge_repeats(frames, transcript)
    word_segments = merge_words(transcript, segments)
    plot_alignments(segments, word_segments, waveform[0], emission.shape[0])
    plt.show()
    return word_segments, frame_alignment


def get_emission(waveform):
    # NOTE: this step is essential
    waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)

    emissions, _ = model(waveform)
    emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0].cpu().detach()

    # Append the extra dimension corresponding to the <star> token
    extra_dim = torch.zeros(emissions.shape[0], emissions.shape[1], 1)
    emissions = torch.cat((emissions, extra_dim), 2)
    emission = emissions[0].cpu().detach()
    return emission, waveform


# matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)

model = wav2vec2_model(
    extractor_mode="layer_norm",
    extractor_conv_layer_config=[
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 2),
        (512, 2, 2),
    ],
    extractor_conv_bias=True,
    encoder_embed_dim=1024,
    encoder_projection_dropout=0.0,
    encoder_pos_conv_kernel=128,
    encoder_pos_conv_groups=16,
    encoder_num_layers=24,
    encoder_num_heads=16,
    encoder_attention_dropout=0.0,
    encoder_ff_interm_features=4096,
    encoder_ff_interm_dropout=0.1,
    encoder_dropout=0.0,
    encoder_layer_norm_first=True,
    encoder_layer_drop=0.1,
    aux_num_out=31,
)

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, progress=True, model_dir="ctc_alignment_mling_uroman")
model.load_state_dict(checkpoint)
model.eval()

# transcript = "aber seit ich bei ihnen das brot hole brauch ich viel weniger schulze wandte sich ab die kinder taten ihm leid"
# SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/10349_8674_000087.flac")

transcript = "i had that curiosity beside me at this moment"
SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
print(SPEECH_FILE)

# transcript_file = sys.argv[1]
# SPEECH_FILE = sys.argv[2]

# from unidecode import unidecode
# import re
# def normalize_uroman(text):
#     text = text.lower()
#     text = text.replace("’", "'")
#     text = re.sub("([^a-z' ])", " ", text)
#     text = re.sub(' +', ' ', text)
#     return text.strip()

# result = []
# for line in open(transcript_file):
#     fields = line.rstrip().split('\t')
#     if len(fields) != 3:
#         print(fields)
#         continue
#     _, _, text = fields
#     text_norm = normalize_uroman(unidecode(text))
#     result.append(text_norm)
# transcript = ' '.join(result)
# print(transcript)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)

emission, waveform = get_emission(waveform)

# Construct the dictionary
# '@' represents the OOV token, '*' represents the <star> token.
# <pad> and </s> are fairseq's legacy tokens, which're not used.
dictionary = {
    "<blank>": 0,
    "<pad>": 1,
    "</s>": 2,
    "@": 3,
    "a": 4,
    "i": 5,
    "e": 6,
    "n": 7,
    "o": 8,
    "u": 9,
    "t": 10,
    "s": 11,
    "r": 12,
    "m": 13,
    "k": 14,
    "l": 15,
    "d": 16,
    "g": 17,
    "h": 18,
    "y": 19,
    "b": 20,
    "p": 21,
    "w": 22,
    "c": 23,
    "v": 24,
    "j": 25,
    "z": 26,
    "f": 27,
    "'": 28,
    "q": 29,
    "x": 30,
    "*": 31,
}
assert len(dictionary) == emission.shape[1]


# One can follow the following steps to download the uroman romanizer and use it to obtain normalized transcripts.
# def normalize_uroman(text):
#     text = text.lower()
#     text = text.replace("’", "'")
#     text = re.sub("([^a-z' ])", " ", text)
#     text = re.sub(' +', ' ', text)
#     return text.strip()
#
# echo 'aber seit ich bei ihnen das brot hole brauch ich viel weniger schulze wandte sich ab die kinder taten ihm leid' > test.txt"
# git clone https://github.com/isi-nlp/uroman
# uroman/bin/uroman.pl < test.txt > test_romanized.txt
#
# file = "test_romanized.txt"
# f = open(file, "r")
# lines = f.readlines()
# text_normalized = normalize_uroman(lines[0].strip())


word_segments, frame_alignment = compute_and_plot_alignments(
    transcript, dictionary, emission, waveform
)
