import numpy as np

def g(z):
  return 1.0 / (1.0 + np.exp(np.array(-z, dtype=np.float128)))