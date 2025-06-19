import numpy as np


def gd_train(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 10) -> np.ndarray:
    """Simple gradient descent trainer used for model stubs."""
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        preds = X.dot(w)
        grad = X.T.dot(preds - y) / len(y)
        w -= lr * grad
    return w


def predict(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X.dot(weights)
