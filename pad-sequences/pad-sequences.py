import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if seqs == []:
        return np.array([], dtype = int).reshape(0,0)
    if max_len == None:
        max_len = max(len(x) for x in seqs)
    result = []
    for s in seqs:
        if len(s) > max_len:
            result.append(s[:max_len])
        else:
            result.append(list(s) + [pad_value] * (max_len - len(s)))
    return np.asarray(result)