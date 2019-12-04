import librosa

def read_song_file(path):

    data, sample_rate = librosa.load(path)

    #Calculate the zero crossing rate
    zero_crossings = librosa.zero_crossings(data, pad=False)

    #Calculate the spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(data, sr=sample_rate)[0]
    
    #Calculate the spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(data, sr=sample_rate)[0]

    #Calculate mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(data, sr=sample_rate)[0]

    #Calculate the chroma frequencies
    hop_length = 512
    chroma = librosa.feature.chroma_stft(data, sr=sample_rate, hop_length=hop_length)

    return zero_crossings, spectral_centroids, spectral_rolloff, mfccs, chroma