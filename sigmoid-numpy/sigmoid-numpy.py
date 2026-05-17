import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    sig = 1/(1+np.exp(-np.asarray(x)))
    return sig