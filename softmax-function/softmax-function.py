import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asarray(x)
    x = x - np.max(x, axis = -1, keepdims = True)
    soft = np.exp(x) / np.sum(np.exp(x), axis = -1, keepdims = True)
    return soft