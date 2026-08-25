import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.array(x)
    result = np.where(x > 0, x, 0)
    return result