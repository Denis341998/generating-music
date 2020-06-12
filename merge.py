import numpy as np
from tqdm import tqdm
import midi_manipulation
import os
import miditomp3

class merge(object):

    def __init__(self):
        pass

    def get_files(self, path="generated"):
        generated = []
        for a, b, c in os.walk(path):
            for f in c:
                generated.append(os.path.join(a, f))

        return generated


    def merge(self, output, midi):
        files = self.get_files()
        songs = np.zeros((0,156))
        for f in tqdm(files):
            try:
                song = np.array(midi.midi_to_note_state_matrix(f))

                #if np.array(song).shape[0] > 10:
                #    print (song)
                #songs.append(song)
                songs = np.concatenate((songs,song))
            except Exception as e:
                raise e

        print( "samples merging ...")

        print( np.shape(songs))
        if(not ".mid" in output):
            output+=".mid"

        midi.note_state_matrix_to_midi(songs, 1, output)

        tomp3 = miditomp3.midi_to_mp3()
        tomp3.midi_to_mp3(output)
        #os.remove(output)
