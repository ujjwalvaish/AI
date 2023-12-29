import numpy as np
from collections import Counter

def calculate_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

class KNN:
    def __init__(self, k) -> None:
        self.k = k
        self.X = None

    def fit(self, X, y):
        self.X = X
        self.y = y
    
    # KNN can be used for both classification and regression
    def predict(self, X_test ,regression = True):
        if regression == True:
            preds = [self.predict_regression(x) for x in X_test]
        else:
            preds = [self.predict_classification(x) for x in X_test]
        return preds 

    def predict_regression(self, x):
        distances = np.array([calculate_distance(point, x) for point in self.X])
        closest_points = np.argsort(distances)[:self.k]
        closest_labels = self.y[closest_points]
        return (np.sum(closest_labels))/self.k
    
    def predict_classification(self, x):
        distances = np.array([calculate_distance(point, x) for point in self.X])
        closest_points = np.argsort(distances)[:self.k]
        closest_labels = self.y[closest_points]
        # print(f"Most common - {Counter(closest_labels).most_common()[0][0]}")
        return Counter(closest_labels).most_common()[0][0]


