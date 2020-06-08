import os
import miditomp3
import numpy as np
# import msgpack
import midi_manipulation
import neuralnet
import sys
import merge
from tqdm import tqdm
import argparse
import tests


class main(object):

    def __init__(self):
        pass

    def get_files(self, path, midi):
        songs = []

        for a, b, c in os.walk(path):
            for f in tqdm(c):
                song = os.path.join(a, f)
                try:
                    n = np.array(midi.midiToNoteStateMatrix(song))
                    songs.append(n)
                except Exception as e:
                    print("error: {} on file: {}".format(e, f))

        return songs

    def delete_duplicate(self, mass):
        out = np.array([0])
        out = np.delete(out, 0)

        for x in mass:
            if not x in out:
                out = np.append(out, x)
                out.append(x)
        return out


if __name__ == "__main__":

    # tests.start()
    # По умолчанию зададим директорию, на тот случай, если аргументы командной строки пусты
    path = "input_midi"
    output = "output"
    start_test = ""

    parser = argparse.ArgumentParser(description='Generate Piano music.')
    parser.add_argument('-d', '--directory', type=str, required=False, metavar='',
                        help='the music to be trained should be inside this dir, if not specified, input_midi will be the input')
    parser.add_argument('-o', '--output', type=str, required=False, metavar='',
                        help='Output music as an .mp3 file, if not specified, will be output.mp3')
    parser.add_argument('-t', '--test', type=str, required=False, metavar='',
                        help='the tests are run. To use it write "start"')
    args = parser.parse_args()

    if (args.directory):
        path = args.directory
        print(path)

    if (args.output):
        output = args.output
        print(output)
    if (args.test):
        start_test = args.test
        if (start_test == 'start'):
            tests.start()

    midi = midi_manipulation.midi_manipulation()

    ma = main()

    songs = ma.get_files(path, midi)
    # midi.mass_check = set(midi.mass_check)
    if len(songs) == 0:
        print('error: the directory is empty')
    else:
        # print(songs)
        if (not os.path.isdir("generated")):
            os.mkdir("generated")

        # print("11111111111111111111111111111111111111111111111111111112")
        neur = neuralnet.neuralnet()
        neur.train(songs, midi, start_test)
        mer = merge.merge()
        mer.merge(output, midi)