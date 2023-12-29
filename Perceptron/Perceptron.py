import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Single layer NN
class Perceptron:
    def __init__(self, units = 1, epochs = 100, lr = 0.01, verbose = True):
        self.units = units
        self.epochs = epochs
        self.lr = lr
        self.weights = None
        self.bias = None
        self.verbose = verbose

    def fit(self, X, y):
        self.X = X
        print(f"X.shape = {X.shape}")
        n_samples, n_features = X.shape
        self.y = y
        print(f"y.shape = {y.shape}")
        self.bias = np.zeros((1, self.units))
        print(f"self.bias.shape = {self.bias.shape}")
        self.weights = np.random.randn(n_features * self.units).reshape(n_features, self.units)
        print(f"self.weights.shape = {self.weights.shape}")
        for _ in range(self.epochs):
            linear_pred = np.dot(X, self.weights) + self.bias
            print(f"linear_pred.shape = {linear_pred.shape}")
            pred = sigmoid(linear_pred)
            dw = (2/n_samples) * np.dot(X.T, (pred - y)) 
            db = (2/n_samples) * np.sum(pred - y)
            
            # Update the bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            if self.verbose:
                y_pred = np.dot(self.weights, X) + self.bias
                pred = sigmoid(y_pred)
                print(f"Accuracy = {np.sum(pred == y )/ len(y)}")

    def predict(self, X_pred):
        linear_pred = np.dot(self.weights, X_pred[1]) + self.bias
        pred = sigmoid(linear_pred)
        return pred
