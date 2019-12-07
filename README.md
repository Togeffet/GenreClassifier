```
================================================================================
      _____                        _____ _               _  __ _           
     / ____|                      / ____| |             (_)/ _(_)          
    | |  __  ___ _ __  _ __ ___  | |    | | __ _ ___ ___ _| |_ _  ___ _ __ 
    | | |_ |/ _ \ '_ \| '__/ _ \ | |    | |/ _` / __/ __| |  _| |/ _ \ '__|
    | |__| |  __/ | | | | |  __/ | |____| | (_| \__ \__ \ | | | |  __/ |   
     \_____|\___|_| |_|_|  \___|  \_____|_|\__,_|___/___/_|_| |_|\___|_|                                             
--------------------------------------------------------------------------------
            By Alec Waichunas, Caleb Cassady, and Franklin Fanelli
                    Fall 2019, SIU CS 437 - Dr. Hoxha
================================================================================
```
# Summary

This program uses machine learning models as a means to predict the genre that
a song file falls into. 
The first model used to make these predictions is a simple logistic regression 
model trained by a gradient descent algorithm. The implementation of this model 
and the gradient descent algorithm were all coded as part of the development of 
this project.
The second model used to make genre predictions is a Dense Neural Network of
five layers. This network was implemented using the Tensorflow library and was
trained on the same data as the first model so that comparisons between the
performance/accuracy of the two models could be made.


# About The Task

Musical genre identification is an interesting and difficult task for machine
learning. For one, songs within a single genre can range across many different
sub-styles, use many different type of instruments, or have vocals from a wide
variety of qualities. In addition, the waveform for a song has many possible
features that one could use for training, so selecting the most distinctive
features and discarding those with relatively little weight is an important but
intricate problem.

We approached this task by exploring previous research and other projects within
the realm of audio classification. Using this information, we identified five
primary features that we could use for training a classification model:

    1. Zero Crossing Rate - The rate of sign-changes along an audio signal. When
                            a waveform is visualed, this can be seen as the
                            number of times the wave crosses the zero-pressure
                            mark.
            
    2. Spectral Centrid - The weighted mean of the frequencies present in the
                          sound. (Could be seen as the wave's "center of mass")

    3. Spectral Rolloff - The frequency below with a specified percentage of
                          total spectral energy lies. Roughly represents the
                          shape of the audio signal.
    
    4. Mel-Frequency Cepstral Coefficients - Describes the shape of the spectral
                                             envelope.
                                             
    5. Chroma Frequencies - Measures the strength of frequencies grouped within
                            each of the 12 distinct musical semitones (chroma)
                            of the musical octave.

These features are extracted from the middle 30 seconds of each song to produce
the feature data.


# Software Requirements

To run this program, the following libraries are required:

    1. Librosa (https://librosa.github.io/librosa/)
        Desc: "LibROSA is a python package for music and audio analysis. It 
               provides the building blocks necessary to create music 
               information retrieval systems."
        Install: 'pip install librosa'
    
    2. Tensorflow (https://www.tensorflow.org/)
        Desc: "TensorFlow is an end-to-end open source platform for machine 
               learning. It has a comprehensive, flexible ecosystem of tools, 
               libraries and community resources that lets researchers push the 
               state-of-the-art in ML and developers easily build and deploy ML 
               powered applications."
        Install: 'pip install tensorflow'


# Usage Instructions
Run the main python file once all dependencies are installed. By default, the program will use the features and model files we've included. Using the trained models and data we’ve created you should be able to get the same results we’ve gotten here. If you want to run this on new data, make sure to delete the files named: 

test_features.npy
train_features.npy
Trained_models.npy

This will force the python file to create new data based on whatever genres and songs are included in the music/ folder.

# Folder Contents
main.py - this is the python file to run. It calls the functions to create training data if it isn’t there, it runs gradient descent on the models if the trained model matrix isn’t there, and then shows the 

createdata.py - this is the function that is called to create the models and featureset

g.py - the sigmoid function used in logistic regression

printgenreforint.py - has a function to print out the name of directory based on the int given

train_features.npy - These are the features generated using the training data set of songs

test_features.npy - Features generated using the test songs

trained_models.npy - Trained theta matrix generated from our gradient descent file

music/test &
music/train - Not included in the github, so please download from the link we provided (for those of you who aren't Dr. Hoxha, please create your own music folders).
These are set up like the following:
```
    music/
        test/
            rap/
                song_file_1.mp3
                song_file_2.mp3
            country/
                song_file_3.mp3
            ...
```



## Development Process
Genre Classifier is a project that learns genres of songs based on ~20 songs from each of the 8 genres. It’s written in Python and features multithreading and a little bit of TensorFlow. 

### Is there anything out there like it? Is it special?
There are things out there that accomplish what we want. Spotify uses something like this to sort songs by many different genres, here is an example of what others have done and more about what Spotify does here: https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8

What makes ours special is that this was made by some students who were not-yet jaded or burnt out and had many stars in their eyes while making this. Working with machine learning principles we learned in class and doing more research on our own to try to get this to be more accurate was really satisfying. Although this isn’t the fastest program (or the most accurate) it was made with love. (Sorry if this sounds cheesy, it’s really the only reason someone would want to use our program over something made by pros).

### Original Ideas
At first, we wrote our own gradient descent and logistic regression algorithms, using what we learned in class. We created trained models containing 8 different gradient-descented-models, that could predict if 1: it was the genre or 0: it wasn’t the genre.

Using these models, we ran them on each song, generating prediction matrices that were 8 rows by 1 column, containing the probability of what the models thought of the current song. Then, we checked the max number in this matrix and saw what model had the highest prediction.

### Issues
Doing it this way gave us quite a few issues, usually by guessing the same genre for every song. This could be the fault of a lot of things:
We had different numbers of training data for each of the genres. I.e. country had 20 songs but rap had 15.
We had a very small amount of data. If we had access to thousands of mp3 files, we’d be able to train these models to have a much higher accuracy.
Songs can belong to more than one genre, some people can’t even do what this is trying to do. There are many songs that can be classified as “both rock and alternative” or even a “mix of country and rap.”
Data is too complex to go through with just one model




### Cutting Down on Genres
As having 8 genres gave us mostly-incorrect predictions, we tried taking some genres out to see if it could correctly distinguish between 2 genres. Our algorithm worked much better doing this, and was able to correctly predict around ~50% of the songs.
When having 3 genres available to create models, our hand-coded logistic regression algorithm gets 23 songs guessed correctly out of 50.



### Trying out TensorFlow
Switching to Tensorflow, we found varying results on each run. Sometimes, we got accuracy of 100%, other times we got 0%, it really depended on how it was feeling. We tried different optimization functions included with TensorFlow. Out of SGD (Stochastic Gradient Descent), Adam, Adamax, and Adadelta, we found that Adadelta worked the best. We also experimented with learning rate and rho values (decay rate), and found that 0.02 and rho of 1.5 worked best.

We've included both our own algorithm and TenorFlow's algorithms in the project, so you can see how ours stacks up against professional machine learning stuff.
