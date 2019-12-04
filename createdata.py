import librosa
import os
import numpy as np

def read_song_file(path):

    data, sample_rate = librosa.load(path)

    total_samples = np.size(data)
    total_seconds = total_samples / sample_rate

    middle_samples = total_samples / 2

    #From position is 15 seconds before the middle
    from_pos = middle_samples - (15 * sample_rate)
    #To position is 15 seconds after the middle
    to_pos = middle_samples + (15 * sample_rate)

    #Extract data 
    middle_data = data[from_pos:to_pos]
    
    print("{} Total Samples ({}s)".format(total_samples, total_seconds))
    print("Middle section has {} total samples ({}s)".format(to_pos - from_pos, (to_pos - from_pos) / sample_rate))

    #Calculate the zero crossing rate
    zero_crossings = librosa.zero_crossings(y=middle_data, pad=False)
    zero_crossings = np.count_nonzero(zero_crossings)
    print(zero_crossings)

    #Calculate the spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=middle_data, sr=sample_rate)
    spectral_centroids.flatten()
    print(spectral_centroids.shape)
    
    #Calculate the spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=middle_data, sr=sample_rate)
    spectral_rolloff.flatten()
    print(spectral_rolloff.shape)

    #Calculate mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=middle_data, sr=sample_rate)
    mfccs.flatten()
    print(mfccs.shape)

    #Calculate the chroma frequencies
    hop_length = 512
    chroma = librosa.feature.chroma_stft(y=middle_data, sr=sample_rate, hop_length=hop_length)
    chroma.flatten()
    print(chroma.shape)

    end_data = np.concatenate( (spectral_centroids, spectral_rolloff, mfccs, chroma) , axis=None )
    np.insert(end_data, 0, zero_crossings)

    return end_data

def convert_to_features(path):

    print("Converting files in {} to feature data file.".format(path))

    #Get directory list
    dirs = os.listdir(path)
    dirs.sort()

    class_dict = dict()
    id_counter = 0

    #Create a dict used to map genre names to integer values
    for directory in dirs:
        print("\tFound directory for {} genre".format(directory))
        class_dict[directory] = id_counter
        id_counter += 1

    #Count the total number of files (used to set numpy array size)
    num_songs = 0
    for directory in dirs:
        if(os.path.isdir(path + "/" + directory)):
            song_files = os.listdir(path + "/" + directory)
            num_songs += len(song_files)

    #Create matrix
    data_matrix = np.empty([num_songs, 43929])

    #Start converting the files
    entry_row = 0
    for directory in dirs:

        if not os.path.isdir(path + "/" + directory):
            continue

        #List files in each directory
        song_files = os.listdir(path + "/" + directory)

        for sfile in song_files:

            #Full name relative to working dir
            full_name = path + "/" + directory + "/" + sfile
            print("\tGetting features for " + full_name)

            #Get features and insert them at the given row in the array
            data_matrix[entry_row] = np.insert(read_song_file(full_name), 0, class_dict[directory])

            #Increment the insert index
            entry_row += 1
    
    np.save(path + "/features", data_matrix)
    return

def read_feature_data(path):
    return np.load(path + "/features.npy")