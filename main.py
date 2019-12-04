import createdata
import gradientdescent as gd
import os

train_data = "music/train"
test_data = "music/test"

if not os.path.exists("music/train/features.npy"):
  createdata.convert_to_features(train_data)

X_and_y_matrix = createdata.read_feature_data(train_data)

theta_matrix = gd.gradient_descent(X_and_y_matrix[:, 1:], X_and_y_matrix[:, 0], 0.07)
print(theta_matrix)

print(theta_matrix.shape)