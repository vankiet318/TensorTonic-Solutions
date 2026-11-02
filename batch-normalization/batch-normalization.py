import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray(x)
    
    gamma = np.asarray(gamma)
    
    beta = np.asarray(beta)

    if x.ndim == 2:
        axis = (0,)
        shape = (1, -1)
    else:
        axis = (0, 2, 3)
        shape = (1, -1, 1, 1)
    
    mean_x = np.mean(x, axis = axis, keepdims = True)
    
    variance = np.mean((x - mean_x) ** 2, axis = axis, keepdims = True)

    gamma = gamma.reshape(shape)
    beta = beta.reshape(shape)
    x_normalize = (x - mean_x) / np.sqrt(variance + eps)

    y = gamma*x_normalize + beta
    
    return y