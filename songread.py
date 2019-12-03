import librosa

def read_song_file(path):
    data, sampleRate = librosa.load(path)
    return data, sampleRate