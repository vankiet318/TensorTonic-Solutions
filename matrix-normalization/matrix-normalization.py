import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    matrix = np.array(matrix)
    if norm_type == "l1":
        norm = np.sum(np.abs(matrix), axis = axis, keepdims = True)
    elif norm_type == "l2":
        norm = np.sqrt(np.sum(np.square(matrix),axis = axis, keepdims = True))
    else:
        norm = np.max(np.abs(matrix),axis = axis, keepdims = True)
    safe_norm = np.where(norm == 0, 1.0, norm)
    return matrix / safe_norm