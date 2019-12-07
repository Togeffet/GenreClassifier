<pre>================================================================================
      _____                        _____ _               _  __ _           
     / ____|                      / ____| |             (_)/ _(_)          
    | |  __  ___ _ __  _ __ ___  | |    | | __ _ ___ ___ _| |_ _  ___ _ __ 
    | | |_ |/ _ \ '_ \| '__/ _ \ | |    | |/ _` / __/ __| |  _| |/ _ \ '__|
    | |__| |  __/ | | | | |  __/ | |____| | (_| \__ \__ \ | | | |  __/ |   
     \_____|\___|_| |_|_|  \___|  \_____|_|\__,_|___/___/_|_| |_|\___|_|                                             
--------------------------------------------------------------------------------
            By Alec Waichunas, Caleb Cassady, and Franklin Fanelli
                    Fall 2019, SIU CS 437 - Dr. Hoxha
================================================================================</pre
%%%%%%%%%%%%%%%%%%%%%%%%%        SUMMARY        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


%%%%%%%%%%%%%%%%%%%%%%        ABOUT THE TASK        %%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


%%%%%%%%%%%%%%%%%%%%%       SOFTWARE REQUIREMENTS        %%%%%%%%%%%%%%%%%%%%%%%

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


%%%%%%%%%%%%%%%%%%%%%%       USAGE INSTRUCTIONS        %%%%%%%%%%%%%%%%%%%%%%%%%

