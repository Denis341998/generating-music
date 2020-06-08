import unittest
import numpy as np
import midi_manipulation
import neuralnet

def start():
    #print("Start tests")
    # test decode_midi in neuralnet
    decode_midi_typical_test()
    decode_midi_empty_mass()
    decode_midi_one_song_some_notes()
    decode_midi_some_song()
    decode_midi_one_track_in_song()
    decode_midi_one_active_note_in_track()
    decode_midi_one_song_one_track_some_notes()

    #test check_song in midi_manipulation
    check_song_typical_test()
    check_song_error_in_middle()
    check_song_big_step()
    check_song_zero_step()
    check_song_same_data_in_different_order()
    check_song_mass_check_more_song_good()
    check_song_mass_check_more_song_bad()
    check_song_mass_check_less_song()
    check_song_some_track()
    check_song_some_track_in_middle()
    check_song_real_data_good()
    check_song_real_data_error_in_end()
    check_song_real_data_error_in_middle()

    #test error in neuralnet
    error_typical_test()
    error_empty_mass()
    error_zero_error()
    error_all_bad()
    error_real_data_little_error()
    error_real_data_all_bad()
    error_real_data_big_error()

    print('Unit tests work!')


def decode_midi_typical_test():
    neur = neuralnet.neuralnet()
    song = [[[1,1,0,0,1], [1,0,0,0,0], [0,1,0,1,1]], [[1,0,0,0,1], [1,0,1,0,1], [1,1,1,1,1]]]
    mass = [0,1,4,0,1,3,4,0,4,0,2,4,0,1,2,3,4]
    res = neur.decode_midi(song)
    assert mass == res, "Typical test error"

def decode_midi_empty_mass():
    neur = neuralnet.neuralnet()
    song = [[[]]]
    mass = []
    res = neur.decode_midi(song)
    assert mass == res, "Empty mass error"

def decode_midi_one_song_some_notes():
    neur = neuralnet.neuralnet()
    song = [[[1,1,0,0,1], [1,0,0,0,0], [0,1,0,1,1]]]
    mass = [0,1,4,0,1,3,4]
    res = neur.decode_midi(song)
    assert mass == res, "One song some notes error"

def decode_midi_some_song():
    neur = neuralnet.neuralnet()
    song = [[[1,1,0,0,1], [1,0,0,0,0], [0,1,0,1,1]], [[1,0,0,0,1], [1,0,1,0,1], [1,1,1,1,1]], [[1,0,0,0,1], [1,0,1,0,1], [1,1,1,1,1]], [[1,0,0,0,1], [1,0,1,0,1], [1,1,1,1,1]], [[1,0,0,0,1], [1,0,1,0,1], [1,1,1,1,1]]]
    mass = [0,1,4,0,1,3,4,0,4,0,2,4,0,1,2,3,4,0,4,0,2,4,0,1,2,3,4,0,4,0,2,4,0,1,2,3,4,0,4,0,2,4,0,1,2,3,4]
    res = neur.decode_midi(song)
    assert mass == res, "Some song error"

def decode_midi_one_track_in_song():
    neur = neuralnet.neuralnet()
    song = [[[1,1,0,0,1]], [[1,1,1,1,1]]]
    mass = [0,1,4,0,1,2,3,4]
    res = neur.decode_midi(song)
    assert mass == res, "One track in song error"

def decode_midi_one_active_note_in_track():
    neur = neuralnet.neuralnet()
    song = [[[1], [1,0,0,0,0], [0,0,0,1,0]], [[1,0], [0,1], [0,0,0,0,1]]]
    mass = [0,0,3,0,1,4]
    res = neur.decode_midi(song)
    assert mass == res, "One active note in track error"

def decode_midi_one_song_one_track_some_notes():
    neur = neuralnet.neuralnet()
    song = [[[0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,0,1,0,0,0,0,1,1,0,1,0,0,1]]]
    mass = [1,5,6,9,10,13,14,15,18,23,24,26,29]
    res = neur.decode_midi(song)
    assert mass == res, "One song one track some notes error"


def check_song_typical_test():
    song = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 8]])
    mass_check = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    n = 1
    song_res = np.array([[1, 2, 3], [4, 5, 6]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Typical test error"

def check_song_error_in_middle():
    song = np.array([[1,2,3], [4,5,5],[7,8,9]])
    mass_check = np.array([[1,2,3], [4,5,6], [7,8,9]])
    n = 1
    song_res = np.array([[1,2,3], [7,8,9]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Error in middle"

def check_song_big_step():
    song = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 8]])
    mass_check = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    n = 4
    song_res = np.array([[1,2,3], [4,5,6], [7,8,8]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Big step error"

def check_song_zero_step():
    song = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 8]])
    mass_check = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    n = 0
    song_res = np.array([[1,2,3], [4,5,6], [7,8,8]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Zero step error"

def check_song_same_data_in_different_order():
    song = np.array([[7,8,9], [4,5,6], [1,2,3]])
    mass_check = np.array([[1,2,3], [4,5,6], [7,8,9]])
    n = 2
    song_res = np.array([7,8,9])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Same data in a different order error"

def check_song_mass_check_more_song_good():
    song = np.array([[1,2,3], [4,5,6], [7,8,8]])
    mass_check = np.array([[1,2,3], [4,5,6], [7,8,9], [1,2,3], [4,5,6], [7,8,8]])
    n = 1
    song_res = np.array([[1,2,3], [4,5,6], [7,8,8]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "mass_check more than song good error"

def check_song_mass_check_more_song_bad():
    song = np.array([[1,2,3], [4,5,6], [7,8,8]])
    mass_check = np.array([[1,2,3, 4,5,6], [7,8,9, 1,2,3], [4,5,6, 7,8,7]])
    n = 2
    song_res = np.array([[1,2,3], [4,5,6]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "mass_check more than song bad error"

def check_song_mass_check_less_song():
    song = np.array([[1,2,3], [4,5,6], [7,8,8]])
    mass_check = np.array([[1,2,3, 4,5,6]])
    n = 2
    song_res = np.array([[1,2,3], [4,5,6]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "mass_check less than song error"

def check_song_some_track():
    song = np.array([[1,2,3], [4,5,6], [7,7,7], [8,8,8]])
    mass_check = np.array([[1,2,3,4,5,6], [7,8,9,1,2,3], [4,5,6,7,8,8]])
    n = 2
    song_res = np.array([[1,2,3], [4,5,6]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Some track error"

def check_song_some_track_in_middle():
    song = np.array([[1,2,3], [4,5,5], [7,8,8], [8,8,8]])
    mass_check = np.array([[1,2,3,4,5,6], [7,8,9,1,2,3], [4,5,6,7,8,8]])
    n = 2
    song_res = np.array([1,2,3])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Some track in middle error"

def check_song_real_data_good():
    song = np.array([[0,1,1,0,1], [1,1,1,1,1], [1,0,1,0,1]])
    mass_check = np.array([[0,1,1,0,1,1,1,1,1,1], [1,1,1,1,1,1,0,1,0,1], [0,1,0,1,0,1,0,0,0,1]])
    n = 2
    song_res = np.array([[0,1,1,0,1], [1,1,1,1,1], [1,0,1,0,1]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Real data good error"

def check_song_real_data_error_in_end():
    song = np.array([[0,1,1,0,1], [1,1,1,1,1], [1,0,1,0,0]])
    mass_check = np.array([[0,1,1,0,1,1,1,1,1,1], [1,0,1,0,1,0,1,0,1,0], [1,0,0,0,1,0,1,1,1,0]])
    n = 2
    song_res = np.array([[0,1,1,0,1], [1,1,1,1,1]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Real data error in end"

def check_song_real_data_error_in_middle():
    song = np.array([[0,1,1,0,1], [0,1,1,1,0], [1,0,1,0,1]])
    mass_check = np.array([[0,1,1,0,1], [1,1,1,1,1,1,0,1,0,1], [0,1,0,1,0,1,0,0,0,1], [0,1,1,1,0,1,0,1,0,1]])
    n = 2
    song_res = np.array([[0,1,1,0,1],  [1,0,1,0,1]])
    midi = midi_manipulation.midi_manipulation()
    song = midi.check_song(song, mass_check, n)
    assert np.all(song_res == song), "Real data error in middle"


def error_typical_test():
    test = np.array([[1,2,3],[2,3,2],[4,5,7]])
    train  = np.array([[2,3,2],[1,1,1],[4,5,6]])
    loss_res = 5.67
    neur = neuralnet.neuralnet()
    loss = neur.error(test, train)
    assert loss_res == round(loss,2), "Error func typical test"

def error_empty_mass():
    test = np.array([])
    train  = np.array([])
    loss_res = 0.0
    neur = neuralnet.neuralnet()
    loss = neur.error(test, train)
    assert loss_res == loss, "Error func empty array"

def error_zero_error():
    test = np.array([[1,2,3],[2,3,1],[4,5,6]])
    train  = np.array([[1,2,3],[2,3,1],[4,5,6]])
    loss_res = 0.0
    neur = neuralnet.neuralnet()
    loss = neur.error(test, train)
    assert loss_res == loss, "Error func zero error"

def error_all_bad():
    test = np.array([[1,2,3],[2,3,1],[4,5,6]])
    train  = np.array([[2,1,4],[3,1,3],[5,6,4]])
    loss_res = 9.0
    neur = neuralnet.neuralnet()
    loss = neur.error(test, train)
    assert loss_res == loss, "Error func all bad"

def error_real_data_little_error():
    test = np.array([[1,0,0,1,0,1,0,1,1,0], [0,0,0,1,0,1,0,1,1,1]])
    train  = np.array([[1,0,0,0,0,1,0,1,1,0], [0,0,0,0,0,1,0,1,1,1]])
    loss_res = 0.4
    neur = neuralnet.neuralnet()
    loss = neur.error(test, train)
    assert loss_res == loss, "Error func real data little error"

def error_real_data_all_bad():
    test = np.array([[1,0,0,1,0,1,0,1,1,0], [0,0,0,1,0,1,0,1,1,1]])
    train  = np.array([[0,1,1,1,1,0,1,0,0,1], [1,1,1,1,1,0,1,0,0,0]])
    loss_res = 3.6
    neur = neuralnet.neuralnet()
    loss = neur.error(test, train)
    assert loss_res == loss, "Error func real data all bad"

def error_real_data_big_error():
    test = np.array([[1,0,0,1,0,1,0], [0,0,0,1,0,1,0], [0,1,0,1,1,0,1], [1,1,1,1,1,1,1], [0,0,0,0,0,0,0], [1,0,1,0,1,0,1], [0,1,0,1,0,1,0]])
    train  = np.array([[0,1,0,1,0,0,1], [1,1,1,0,1,1,1], [1,0,1,0,0,0,1], [0,0,0,0,0,0,0], [1,1,1,0,0,1,1], [0,1,1,0,0,1,0], [1,0,0,1,0,0,1]])
    loss_res = 28.57
    neur = neuralnet.neuralnet()
    loss = neur.error(test, train)
    assert loss_res == round(loss,2), "Error func real data big error"
