import createdata
import threading
import numpy as np
from g import g
import matplotlib.pyplot as plt
from printgenreforint import genretostring

def gradient_descent(X, Y, alpha, show_graph):
  print("Running gradient descent...")

  X = np.array(X)
  Y = np.array(Y)

  Y = Y.reshape(Y.shape[0], 1)

  samples, features = X.shape

  print("Normalizing input features...")

  # normalize X values
  for i in range (0, features):
    xavg = np.average(X[:,i])
    xptp = np.ptp(X[:,i]) #ptp = "peak to peak", used to get the range of values
    X[:,i] = np.subtract(X[:,i], xavg)
    X[:,i] = np.divide(X[:,i], xptp)

  print("Features normalized.")
  
  # Add extra 1s column to the front
  X = np.hstack((np.ones((samples, 1)), X))

  # Get the number of samples and features again after adding the column
  samples, features = X.shape

  #Get the number of genres
  num_genres = int(np.ptp(Y, axis=0)[0]) + 1

  trained_models = np.ndarray( (features, num_genres) )

  threads = []
  for k in range(0, num_genres):
    thread = threading.Thread(target=train_class_model, args=(k, features, samples, X, Y, alpha, trained_models))
    thread.start()
    threads.append(thread)

  for t in threads:
    t.join()

  if show_graph: plt.show()

  return trained_models

def train_class_model(genre_id, features, samples, X, Y, alpha, trained_models):
  print("Creating model for " + genretostring(genre_id))

  # Create the Y matrix for identifying genre i
  #currentY = np.subtract(Y, i - 1)

  current_genre_indicies = (Y == genre_id)
  currentY = np.zeros(Y.shape)
  currentY[current_genre_indicies]  = 1

  # Initialize theta randomly and a matrix for storing the gradient
  theta_matrix = np.random.rand(features, 1)
  derivative_matrix = np.ones((features, 1))

  # Start gradient descent loop
  iteration = 1
  converged = False
  while not converged:

    h_theta = g(np.matmul(X, theta_matrix))

    for i in range(0, features):
      derivative_matrix[i] = (1 / float(samples)) * np.sum( np.multiply(h_theta - currentY, np.reshape(X[:,i], (samples, 1))) )

      # Just keeping this here, gives the same result as above just doing 
      # print(1 / float(samples)) * np.matmul(np.transpose(np.subtract(h_theta, Y)), X[:,i])

    max_derivative = np.max(derivative_matrix)
    print("Genre: " + genretostring(genre_id) + ", Iteration: " + str(iteration) + ", max derivative: "+ str(max_derivative))

    theta_matrix = theta_matrix - (alpha * derivative_matrix)

    error = 1 / float(2 * samples) * np.sum(np.power(np.subtract(h_theta, currentY), 2))
    iteration += 1

    #Check for convergence
    if all (abs(i) <= 0.001 for i in derivative_matrix[1:]):
      converged = True

  trained_models[:,genre_id] = np.reshape(theta_matrix, (features))
