import createdata
import gradientdescent as gd
import os
import numpy as np
from g import g

# Get rid of the pesky warning for us lowly python 2 users

train_dir = "music/train"
test_dir = "music/test"

#Check to see if we have already processed the song files
#and created the training data.
if not os.path.exists("train_features.npy"):
  createdata.convert_to_features(train_dir, "train_features")

if not os.path.exists("test_features.npy"):
  createdata.convert_to_features(test_dir, "test_features")

#Load training/test data
train_data = createdata.read_feature_data("train_features.npy")
test_data = createdata.read_feature_data("test_features.npy")

#Create variables for number of test/training samples
num_train_samples = train_data.shape[0]
num_test_samples = test_data.shape[0]

#Check to see if the models have already been created
if not os.path.exists("trained_models.npy"):
  trained_models = gd.gradient_descent(train_data[:, 1:], train_data[:, 0], 2, True)
  np.save("trained_models", trained_models)

#Load trained models
trained_models = np.load("trained_models.npy")

#Format test data into samples and expected results
test_X = test_data[:, 1:]
test_X = np.hstack( (np.ones((num_test_samples, 1)), test_X) )
test_Y = test_data[:, 0]

#Get relevant variables
features, samples = test_data.shape
num_genres = int(np.ptp(test_Y))

#Create an array to store our estimates for the 
estimates = np.ndarray((num_genres, samples))

total_guesses = 0
correct_guesses = 0
for i in range(0, test_X.shape[0]):
  sample = test_X[i]
  #sample = np.reshape(sample, (features, 1))

  prediction = np.matmul(np.transpose(trained_models), np.reshape(sample, (sample.shape[0], 1)))

  predicted_genre = np.argmax(prediction, axis=0)
  if predicted_genre == test_Y[i]:
    print("Correctly predicted the genre (" + str(predicted_genre) + ") for song " + str(i))

    total_guesses += 1
    correct_guesses += 1
    
  else:
    print("Incorrect :( Correct genre was " + str(test_Y[i]) + " and we guessed " + str(predicted_genre))

    total_guesses += 1

  # prediction = g(prediction)

print("We guessed " + str(correct_guesses) + " correctly out of " + str(total_guesses))