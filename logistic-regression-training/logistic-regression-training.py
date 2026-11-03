import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n, m = X.shape
    w = np.zeros(m)
    b = 0
    for i in range(steps):
        y_pred = _sigmoid(X @ w +b)
        grad_w = np.dot(X.T, (y_pred - y)) / n
        grad_b = np.mean(y_pred - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return (w,b)
        