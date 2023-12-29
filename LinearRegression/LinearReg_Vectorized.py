import numpy as np

# Same as the other class!
class LinearReg_Vectorized:
    def __init__(self, epochs = 100 , lr = 0.01, verbose = True):
        self.epochs = epochs
        self.lr = lr
        # Do not know the dim of weight, and bias just follows despite knowing dim
        self.weights = None
        self.bias = None
        self.verbose = verbose

    def fit(self, X, y):
        # need dim of x to be n_samples * n_features
        # need dim of y to be n_samples * 1
        n_samples, n_features = X.shape
        y = y.reshape((n_samples, 1))
        # now we know the dims of weights (n_features, 1) and bias (1,1)
        self.weights = np.zeros(n_features).reshape(n_features, 1)
        self.bias = np.zeros(1).reshape(1,1) # random does not matter as much

        for _ in range(self.epochs):
            pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (pred - y))
            db = (1/n_samples) * np.sum(pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.bias * db

            if self.verbose:
                print(f"RMSE = {(1/n_samples) * (np.sqrt(np.sum((pred - y) ** 2)))}")


    def predict(self, X_test):
        pred = np.dot(X_test, self.weights) + self.bias
        return pred