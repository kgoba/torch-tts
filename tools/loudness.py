from pyebur128 import (
    ChannelType, MeasurementMode, R128State, get_loudness_momentary
)
import soundfile as sf

def get_max_loudness_momentary(filename):
    '''Open the WAV file and get the loudness in momentary (400ms) chunks.'''
    with sf.SoundFile(filename) as wav:
        state = R128State(wav.channels,
                          wav.samplerate,
                          MeasurementMode.MODE_M)

        max_momentary = float('-inf')
        total_frames_read = 0
        for block in wav.blocks(blocksize=int(wav.samplerate / 100)):
            frames_read = len(block)
            total_frames_read += frames_read

            for sample in block:
                state.add_frames(sample, 1)

            # Invalid results before the first 400 ms.
            if total_frames_read >= 0.4 * wav.samplerate:
                momentary = get_loudness_momentary(state)
                print(momentary)
                max_momentary = max(momentary, max_momentary)

import sys
get_max_loudness_momentary(sys.argv[1])
