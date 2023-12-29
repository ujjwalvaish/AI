import sys
sys.path.insert(0, '/Users/ujjwalvaish/Desktop/Tech/Projects/ML Algorithms/DecisionTrees')
from DecisionTrees import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, max_depth = 100, min_split = 2,n_trees = 10, total_features = None):
        self.trees = []
        self.max_depth = max_depth
        self.min_split = min_split
        self.total_features = total_features
        self.n_trees = n_trees


    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth, 
                                total_features = self.total_features, 
                                min_elements = self.min_split)
            X_sample, y_sample = self._make_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    # Each tree will make its predictions for all X_test
    # Then we will combine predictions from all trees and return the most common ones
    def predict(self, X_test): # n_samples, n_features
        predictions = np.array([tree.predict(X_test) for tree in self.trees]) # n_trees, n_samples
        tree_preds = np.swapaxes(predictions, 0, 1) 
        # n_samples, n_trees -> every row consists of n_trees predictions for a sample
        predictions = np.array([self._get_most_common(p) for p in tree_preds])
        return predictions

    def _make_samples(self, X, y):
        num_samples = X.shape[0]
        indices = np.random.choice(num_samples, num_samples, replace=True)
        return X[indices], y[indices]
    
    def _get_most_common(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
