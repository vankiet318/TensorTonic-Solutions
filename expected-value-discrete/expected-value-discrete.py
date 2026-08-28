import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    x = np.asarray(x)
    p = np.asarray(p)
    return np.sum(x * p, dtype = float)