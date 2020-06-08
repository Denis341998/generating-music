import midi
import numpy as np

class midi_manipulation(object):

    def __init__(self):
        self.lowerBound = 24
        self.upperBound = 102
        self.span = self.upperBound - self.lowerBound
        self.midi_count = 1 #оптимальный вариант
        self.mass_check = np.array([0])
        self.mass_check = np.delete(self.mass_check, 0)


    def fill_mass_check(self, statematrix):

        for i in range(self.midi_count-1, len(statematrix)):
            mass_check_tmp = np.array([0])
            mass_check_tmp = np.delete(mass_check_tmp, 0)
            for cnt in range(self.midi_count):
                for j in range(len(statematrix[i])):
                    mass_check_tmp = np.append(mass_check_tmp, int(statematrix[i-cnt][j]))

            ###self.mass_check = np.append(self.mass_check, mass_check_tmp)
            if len(self.mass_check) == 0:
                self.mass_check = np.append(self.mass_check, mass_check_tmp)
            else:
                count = 0
                for i in range(len(self.mass_check)):
                    if np.all(self.mass_check[i] == mass_check_tmp):
                        count += 1
                        break
                if count == 0:
                    self.mass_check = np.vstack((self.mass_check, mass_check_tmp))


        #return mass_check

    def check_song(self, song, mass_check, n):
        #mass_check_tmp = np.array([])
        if n == 0:
            return song
        count = 0

        #song_tmp = np.array(song)
        for i in range(len(song) - 1, n-2, -1):

            mass_check_tmp = np.array([0])
            mass_check_tmp = np.delete(mass_check_tmp, 0)
            for cnt in range(n-1, -1, -1):
                for j in range(len(song[i])):
                    mass_check_tmp = np.append(mass_check_tmp, int(song[i-cnt][j]))#int
            count = 0
            for mass in mass_check:
                if np.all(mass == mass_check_tmp):# == mass_check_tmp:
                    count += 1
            if count == 0:
                song = np.delete(song, i, axis=0)
            '''if np.array(mass_check_tmp) not in np.array(mass_check):
                #song.remove(i)
                del(song[i])'''

        return song

    def mass_to_int(self, mass):
        for elem in mass:
            elem = int(elem)


    def midi_to_note_state_matrix(self, midifile):
        Squash = True
        pattern = midi.read_midifile(midifile)
        timeleft = [track[0].tick for track in pattern]
        posns = [0 for track in pattern]

        statematrix = []
        time = 0

        state = [[0,0] for x in range(self.span)]
        statematrix.append(state)
        condition = True
        while condition:
            if time % (pattern.resolution / 4) == (pattern.resolution / 8):
                oldstate = state
                state = [[oldstate[x][0],0] for x in range(self.span)]
                statematrix.append(state)

            for i in range(len(timeleft)):
                #
                if not condition:
                    break
                while timeleft[i] == 0:
                    track = pattern[i]
                    pos = posns[i]

                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < self.lowerBound) or (evt.pitch >= self.upperBound):
                            pass
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-self.lowerBound] = [0, 0]
                            else:
                                state[evt.pitch-self.lowerBound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2, 4):
                            out =  statematrix
                            condition = False
                            break
                    try:
                        timeleft[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        timeleft[i] = None

                if timeleft[i] is not None:
                    timeleft[i] -= 1

            if all(t is None for t in timeleft):
                break

            time += 1

        S = np.array(statematrix)

        statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
        statematrix = np.asarray(statematrix).tolist()

        self.fill_mass_check(statematrix)  #Не удаляй!

        return statematrix

    def note_state_matrix_to_midi(self, statematrix, name="example"):
        statematrix = np.array(statematrix)
        if not len(statematrix.shape) == 3:
            statematrix = np.dstack((statematrix[:, :self.span], statematrix[:, self.span:]))
        statematrix = np.asarray(statematrix)
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)

        span = self.upperBound-self.lowerBound
        tickscale = 55

        lastcmdtime = 0
        prevstate = [[0,0] for x in range(span)]
        for time, state in enumerate(statematrix + [prevstate[:]]):
            offNotes = []
            onNotes = []
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+self.lowerBound))
                lastcmdtime = time
            #print(offNotes)
            for note in onNotes:
                track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+self.lowerBound))
                lastcmdtime = time
            #print(onNotes)
            prevstate = state

        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        if(not ".mid" in name):
            midi.write_midifile("{}.mid".format(name), pattern)
        else:
            midi.write_midifile(name, pattern)

