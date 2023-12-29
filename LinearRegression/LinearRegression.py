import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.001, epochs = 100) -> None:
        self.lr = lr
        self.iter = epochs
        self.weights = None  # vector
        self.bias = None  # scalar
        pass

    """
    
    """
    def train(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias with zeros - can do random b/w -1 and 1 as well
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iter):
            # Dim 0 of both arguments must be equal, hence transposing X
            y_pred = np.dot(self.weights, X.T) + self.bias
            # Calculating gradients
            """
            Note - Take RMSE as (y_pred - y)**2 / (2*n_samples) 
            to get upward parabola that can be minimized
            """
            db = (1/n_samples)* np.sum(y_pred - y)
            dw = (1/n_samples)* np.dot(X.T, (y_pred - y))

            # Updating weights and bias
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
        print(f"Resulting model is {self.weights}x + {self.bias}")

    def predict(self, X_pred):
        return np.dot(X_pred, self.weights) + self.bias
    
    def calculate_rmse(self, y_pred, y_act):
        return ((y_pred-y_act)**2).mean() / 2


# Test