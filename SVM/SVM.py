import numpy as np

class SVM:
    def __init__(self, lr = 0.001, reg_lambda = 0.01, epochs = 100):
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.weights = None
        self.bias = None
        pass

    def fit(self, X, y):
        # Check if labels are -1 and 1
        y_ = np.where(y < 0, -1, 1)
        # init weights
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Loss function = λ*(w**2) + max(0, 1 - y(w*x - b))
        # dw = 2*λ*w if y(w*x - b) > 1 else 2*λ*w - y*x
        # db = 0 if y(w*x - b) > 1 else y
        for _ in range(self.epochs):
            for idx, x in enumerate(X):
                condition = (y_[idx] * (np.dot(self.weights, x) - self.bias)) 
                if condition > 1:
                    self.weights -= self.lr * 2 * self.reg_lambda * self.weights
                else:
                    self.weights -= self.lr * (2 * self.reg_lambda * self.weights - np.dot(y_[idx], x))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X_test):
        pred = np.dot(self.weights, X_test.T) - self.bias
        return np.sign(pred)
