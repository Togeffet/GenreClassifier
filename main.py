import createdata
import gradientdescent as gd

train_data = "music/train"
test_data = "music/test"

createdata.convert_to_features(train_data)

X_and_y_matrix = createdata.read_feature_data(train_data)

print(gd.gradient_descent(X_and_y_matrix[:, 1:], X_and_y_matrix[:, 0], 0.7))