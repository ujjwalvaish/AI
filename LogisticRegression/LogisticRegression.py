import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self, lr = 0.01, epochs = 1000, threshold = 0.5) -> None:
        self.lr = lr
        self.iter = epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights with zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.iter):
            # Calculate prediction
            linear_pred= np.dot(self.weights, X.T) + self.bias
            pred = sigmoid(linear_pred)

            # Calculate derivatives of the cross entropy loss function
            db = (2/n_samples) * np.sum(pred - y )
            dw = (2/n_samples) * np.dot(X.T, (pred - y))
            # Update the weights and biases
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            # Can uncomment while training to keep a tab on accuracy
            # print(f"Accuracy = {self.evaluate(pred, y)}")

    def predict(self, X_pred):
        linear_pred= np.dot(self.weights, X_pred.T) + self.bias
        pred = sigmoid(linear_pred)
        return [0 if y < self.threshold else 1 for y in pred]

    def evaluate(self, y_pred, y_act):
        # accuracy 
        return np.sum(y_pred == y_act)/ len(y_act)