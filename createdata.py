import librosa
import os
import numpy as np

def read_song_file(path):

    data, sample_rate = librosa.load(path)

    total_samples = np.size(data)
    #total_seconds = total_samples / sample_rate

    middle_samples = total_samples / 2

    #From position is 15 seconds before the middle
    from_pos = middle_samples - (15 * sample_rate)
    #To position is 15 seconds after the middle
    to_pos = middle_samples + (15 * sample_rate)

    #Extract data 
    middle_data = data[int(from_pos):int(to_pos)]

    #Calculate the zero crossing rate
    zero_crossings = librosa.zero_crossings(y=middle_data, pad=False)
    zero_crossings = np.count_nonzero(zero_crossings)

    #Calculate the spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=middle_data, sr=sample_rate)
    spectral_centroids.flatten()
    
    #Calculate the spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=middle_data, sr=sample_rate)
    spectral_rolloff.flatten()

    #Calculate mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=middle_data, sr=sample_rate)
    mfccs.flatten()

    #Calculate the chroma frequencies
    hop_length = 512
    chroma = librosa.feature.chroma_stft(y=middle_data, sr=sample_rate, hop_length=hop_length)
    chroma.flatten()

    end_data = np.concatenate( (spectral_centroids, spectral_rolloff, mfccs, chroma) , axis=None )
    np.insert(end_data, 0, zero_crossings)

    return end_data

def convert_to_features(infile, outfile):

    print("Converting files in {} to feature data file.".format(infile))

    #Get directory list
    dirs = os.listdir(infile)
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
        if(os.path.isdir(infile + "/" + directory)):
            song_files = os.listdir(infile + "/" + directory)
            num_songs += len(song_files)

    #Create matrix
    data_matrix = np.empty([num_songs, 43929])

    #Start converting the files
    entry_row = 0
    for directory in dirs:

        if not os.path.isdir(infile + "/" + directory):
            continue

        #List files in each directory
        song_files = os.listdir(infile + "/" + directory)

        for sfile in song_files:
            #Full name relative to working dir
            full_name = infile + "/" + directory + "/" + sfile
            print("\tGetting features for " + full_name)

            if full_name.endswith(".mp3"):

                #Get features and insert them at the given row in the array
                data_matrix[entry_row] = np.insert(read_song_file(full_name), 0, class_dict[directory])

                #Increment the insert index
                entry_row += 1
    
    np.save(outfile, data_matrix)
    return

def read_feature_data(path):
    return np.load(path)