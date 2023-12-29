import numpy as np
from collections import Counter

class Node:
    def __init__(self, left = None, right = None, feature = None, threshold = None, value = None):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth = 100, min_elements = 2, total_features = None):
        self.max_depth = max_depth 
        self.min_elements = min_elements
        self.num_features = total_features
        self.root = None
    
    def fit(self, X, y):
        self.num_features = X.shape[1] if not self.num_features else min(X.shape[1], self.num_features)
        self.root = self._make_tree(X, y)
    
    """
    1. What feature to split tree on
    2. What value of that feature to split tree on
    3. Stopping criteria

    Note - we split on features, but calculate info gain based on labels
    """
    def _make_tree(self,X, y, depth = 0):
        num_samples, num_feats = X.shape
        num_unique_labels = len(np.unique(y))

        # Stopping criteria
        if num_unique_labels == 1 or num_samples <= self.min_elements or depth >= self.max_depth:
            return Node(value=self._get_most_common(y)) 
        
        # Returns indices of columns to split the data by
        # this also adds randomness to decision trees
        feat_combs = np.random.choice(num_feats, self.num_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_combs)
        
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._make_tree(X[left_idxs,:], y[left_idxs], depth + 1)
        right = self._make_tree(X[right_idxs,:], y[right_idxs], depth + 1)
        # Returns the root node of the main tree after recursively going through all branches
        return Node(left, right, best_feat, best_thresh)



    # Returns the best split
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                info_gain = self._get_info_gain(X_column, y, threshold)
                if info_gain > best_gain:
                    best_gain = info_gain
                    split_idx = feat_idx
                    split_threshold = threshold
        
        return split_idx, split_threshold
    

    # used for leaf node
    def _get_most_common(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    # gets info gain based on a particular split
    def _get_info_gain(self,X_column, y, threshold):
        # entropy(parent) - weighted entropy (child)
        parent_entropy = self._entropy(y)

        # create children
        left_idx, right_idx = self._split(X_column, threshold)

        # there is 0 entropy if we get <= 1 branch
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        # calculated the weighted avg entropy of children
        n = len(y)
        num_left, num_right = len(left_idx), len(right_idx)
        entropy_left, entropy_right = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        weight_left, weight_right = (num_left)/n, (num_right)/n
        child_entropy = entropy_left * weight_left + entropy_right * weight_right
        
        # return info gain
        return parent_entropy - child_entropy


    # Splits a column into left and right based on threshold and returns indices
    def _split(self, X_column, threshold):
        left = np.argwhere(X_column < threshold).flatten()
        right = np.argwhere(X_column >= threshold).flatten()
        return left, right
    
    # calculates entropy for a list of values
    def _entropy(self, y):
        hist = np.bincount(y) # indices are values, and values are frequencies of occurrences
        probs = hist / len(y)
        entropy = - np.sum([p * np.log(p) for p in probs if p > 0])
        return entropy
    
    def predict(self, X_test):
        return np.array([self._traverse_tree(x, self.root) for x in X_test])

    def _traverse_tree(self, x, node : Node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        
