import createdata
import gradientdescent as gd
import os
import numpy as np
from g import g
from printgenreforint import genretostring

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
  trained_models = gd.gradient_descent(train_data[:, 1:], train_data[:, 0], 1, True)
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
top_3_guesses = 0
for i in range(0, test_X.shape[0]):
  sample = test_X[i]
  #sample = np.reshape(sample, (features, 1))

  prediction = np.matmul(np.transpose(trained_models), np.reshape(sample, (sample.shape[0], 1)))

  predicted_genre_matrix = np.column_stack((np.arange(prediction.size), prediction))

  predicted_genre_matrix = predicted_genre_matrix[predicted_genre_matrix[:,1].argsort()][::-1]

  predicted_genre = predicted_genre_matrix[0, 0]

  top_3_genres = predicted_genre_matrix[0:3,0]

  if predicted_genre == test_Y[i]:
    print("Correctly predicted song " + str(i) + " with genre " + genretostring(predicted_genre) + "!")

    correct_guesses += 1
    top_3_guesses += 1

  elif np.isin(test_Y[i], top_3_genres):
    print("Correct genre within top 3 predictions! Song " + str(i) + ", correct genre " + genretostring(test_Y[i]) + ", guessed "),
    for num in top_3_genres:
      print (genretostring(num) + ", "),

    print('')
    top_3_guesses += 1

  else:
    print("Incorrectly predicted song " + str(i) + ", correct genre " + genretostring(test_Y[i]) + ", guessed " + genretostring(predicted_genre))
  
  total_guesses += 1

  # prediction = g(prediction)

print("Correct ratio " + str(correct_guesses) + "/" + str(total_guesses) + " (" + str((correct_guesses / float(total_guesses)) * 100) + "%)")
print("Correct prediction within the top 3 genres " + str(top_3_guesses) + " times (" + str((top_3_guesses / float(total_guesses)) * 100) + "%)") 