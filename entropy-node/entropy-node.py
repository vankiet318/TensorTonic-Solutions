import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    classes = np.unique(y, return_counts = True)[1]
    p = classes / len(y)
    score = - np.dot(p, np.log(p) / np.log(2))
    return score