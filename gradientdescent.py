import numpy as np
from g import g
import matplotlib.pyplot as plt


def gradient_descent(X, Y, alpha, show_graph):
  print("running gradient descent")

  X = np.array(X)
  Y = np.array(Y)

  Y = Y.reshape(Y.shape[0], 1)

  samples, features = X.shape


  # normalize X and Y values
  # This totally changes the output, so there could be something wrong with this
  for i in range (0, features):
    X[:,i] = np.subtract(X[:,i], min(X[:,i]))
    X[:,i] = np.divide(X[:,i], max(X[:,i]))
    
  Y[:,0] = np.subtract(Y[:,0], min(Y[:,0]))
  Y[:,0] = np.divide(Y[:,0], max(Y[:,0]))
  
  
  X = np.hstack((np.ones((samples, 1)), X))

  samples, features = X.shape

  theta_matrix = np.random.rand(features, 1)
  derivative_matrix = np.ones((features, 1))
  
  iteration = 1
  converged = False
  while not converged:
    if all (abs(i) <= 0.001 for i in derivative_matrix):
      converged = True

    h_theta = g(np.matmul(X, theta_matrix))

    for i in range(0, features):
      helpme = np.matmul(np.transpose(np.subtract(h_theta, Y)), X[:,i])
      helpme2 = (1 / float(samples))
      derivative_matrix[i] = (1 / float(samples)) * np.matmul(np.transpose(np.subtract(h_theta, Y)), X[:,i])

    theta_matrix = theta_matrix - (alpha * derivative_matrix)
  
    error = 1 / float(2 * samples) * np.sum(np.power(np.subtract(h_theta, Y), 2));

    if show_graph:
      plt.plot(iteration, error, 'o', color="black")
    

    iteration += 1

  
  print("Gradient descent took " + str(iteration) + " iterations to converge")

  if show_graph: plt.show()

  return theta_matrix