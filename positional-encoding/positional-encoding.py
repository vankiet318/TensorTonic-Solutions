import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pe = np.zeros((seq_len, d_model), dtype = float)
    for pos in range(seq_len):
        for dim in range(d_model):
            if dim % 2 == 0:
                pe[pos][dim] = np.sin( pos / (base ** ( 2 * (dim // 2) / d_model)))
            else:
                pe[pos][dim] = np.cos( pos / (base ** ( 2 * (dim // 2) / d_model)))
    return pe