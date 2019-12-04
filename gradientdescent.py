import numpy as np
from g import g


def gradient_descent(X, Y, alpha):
  print("running gradient descent")
  X = np.array(X)
  Y = np.array(Y)
  
  samples, features = X.shape
  X = np.hstack((np.ones((samples, 1)), X))

  samples, features = X.shape

  theta_matrix = np.random.rand(features, 1)
  derivative_matrix = np.ones((features, 1))

  
  iteration = 1
  converged = False
  while not converged:
    for x in derivative_matrix:
      # If the derivative has converged
      if abs(x) <= 0.001:
        converged = True

    h_theta = g(np.matmul(X, theta_matrix))

    for i in range(0, features):
      kill, me = h_theta.shape
      untransposed = np.subtract(h_theta, Y.reshape(kill, 1))
      transposed = np.transpose(untransposed)

      derivative_matrix[i] = (1 / float(samples)) * np.matmul(transposed, X[:,i])

    theta_matrix = theta_matrix - (alpha * derivative_matrix)
    
    iteration += 1

    if converged:
      break
  
  return theta_matrix