import createdata
import gradientdescent as gd
import os

train_dir = "music/train"
test_dir = "music/test"

if not os.path.exists("train_features.npy"):
  createdata.convert_to_features(train_dir, "train_features")

X_and_y_matrix = createdata.read_feature_data("train_features.npy")

theta_matrix = gd.gradient_descent(X_and_y_matrix[:, 1:], X_and_y_matrix[:, 0], 0.7)
print(theta_matrix)

print(theta_matrix.shape)