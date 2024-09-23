import numpy as np

"""
b = (x.t * x + lambda * i) ** -1 * x.t * y
"""

class Ridge:

    def __init__(self, alpha: float = 1e-2, intercept: bool = False) -> None:
        self.alpha = alpha
        self.intercept = intercept

    def fit(self, X, y) -> None:

        X_input = X

        if self.intercept:
            X_input = np.c_[np.ones((X_input.shape[0], 1)), X_input]

            I = np.identity(X_input.shape[1])
            I[0, 0] = 0

        else:
            I = np.identity(X_input.shape[1])
        
        inner_term = np.linalg.inv(np.dot(X_input.T, X_input) + (self.alpha * I))
        outer_term =  np.dot(X_input.T, y)

        self.weights = np.dot(inner_term, outer_term)

    def predict(self, X_input):

        if self.intercept:
            X_input = np.c_[np.ones((X_input.shape[0], 1)), X_input]

        return np.dot(X_input, self.weights)





