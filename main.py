import createdata
import gradientdescent as gd
import os
import numpy as np

train_dir = "music/train"
test_dir = "music/test"

if not os.path.exists("train_features.npy"):
  createdata.convert_to_features(train_dir, "train_features")

if not os.path.exists("test_features.npy"):
  createdata.convert_to_features(test_dir, "test_features")

train_data = createdata.read_feature_data("train_features.npy")
test_data = createdata.read_feature_data("test_features.npy")

num_train_samples = train_data.shape[0]
num_test_samples = test_data.shape[0]

theta_matrix = gd.gradient_descent(train_data[:, 1:], train_data[:, 0], 0.001, True)

test_X = test_data[:, 1:]
test_X = np.hstack( (np.ones((num_test_samples, 1)), test_X) )
test_Y = test_data[:, 0]

estimates = np.ndarray((num_test_samples, 1))

for i in range(0, test_X.shape[0]):
  sample = test_X[i,:]
  sample = np.reshape(sample, (43929, 1))
  estimate = np.matmul(np.transpose(theta_matrix), sample)
  estimates[i] = estimate
  print("Estimate was {}. Correct value is {}".format(estimates[i], test_Y[i]))