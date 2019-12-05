import createdata
import numpy as np
from g import g
import matplotlib.pyplot as plt


def gradient_descent(X, Y, alpha, show_graph):
  print("Running gradient descent...")

  X = np.array(X)
  Y = np.array(Y)

  Y = Y.reshape(Y.shape[0], 1)

  samples, features = X.shape

  print("Normalizing input features...")

  # normalize X and Y values
  # This totally changes the output, so there could be something wrong with this
  for i in range (0, features):
    xavg = np.average(X[:,i])
    xptp = np.ptp(X[:,i]) #ptp = "peak to peak", used to get the range of values
    X[:,i] = np.subtract(X[:,i], xavg)
    X[:,i] = np.divide(X[:,i], xptp)

  print("Features normalized.")

  # Normalization for output shouldn't be needed, but is included here
  # for quick testing. 
  #Y[:,0] = np.subtract(Y[:,0], min(Y[:,0]))
  #Y[:,0] = np.divide(Y[:,0], max(Y[:,0]))
  
  # Add extra 1s column to the front
  X = np.hstack((np.ones((samples, 1)), X))

  # Get the number of samples and features again after adding the column
  samples, features = X.shape

  #Get the number of genres
  num_genres = int(np.ptp(Y, axis=0)[0])

  trained_models = np.ndarray( (features, num_genres) )

  for k in range(0, num_genres):
    print("Creating model for genre #" + str(k))

    # Create the Y matrix for identifying genre i
    #currentY = np.subtract(Y, i - 1)

    current_genre_indicies = Y == k
    currentY = np.zeros(Y.shape)
    currentY[current_genre_indicies]  = 1

    # Initialize theta randomly and a matrix for storing the gradient
    theta_matrix = np.random.rand(features, 1)
    derivative_matrix = np.ones((features, 1))

    # Start gradient descent loop
    iteration = 1
    converged = False
    while not converged:

      if iteration % 100 == 0: print("Iteration number " + str(iteration))

      h_theta = g(np.matmul(X, theta_matrix))

      for i in range(0, features):
        derivative_matrix[i] = (1 / float(samples)) * np.sum( np.multiply(h_theta - currentY, np.reshape(X[:,i], (samples, 1))) )

        # Just keeping this here, gives the same result as above just doing 
        # print(1 / float(samples)) * np.matmul(np.transpose(np.subtract(h_theta, Y)), X[:,i])

      print(derivative_matrix)

      theta_matrix = theta_matrix - (alpha * derivative_matrix)

      error = 1 / float(2 * samples) * np.sum(np.power(np.subtract(h_theta, currentY), 2))

      if show_graph:
        plt.plot(iteration, error, 'o', color=(1/float(i+1), 1/(features - i), 0.5, 1))
    
      iteration += 1

      #Check for convergence
      if all (abs(i) <= 0.001 for i in derivative_matrix[1:]):
        converged = True

    trained_models[:,k] = np.reshape(theta_matrix, (features))
  
  print("Gradient descent took " + str(iteration) + " iterations to converge")

  if show_graph: plt.show()

  return trained_models