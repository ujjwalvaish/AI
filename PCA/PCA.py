import numpy as np

# Beautifully explained here - https://towardsdatascience.com/eigenvalues-and-eigenvectors-378e851bf372
class PCA:
    def __init__(self, n_dims = 2):
        self.dims = n_dims
        self.mean = None
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X)
        # calc covariance matrix - accepts each column as an observation
        # needs shape of X to be n_features * n_samples
        X_transposed = X.T
        
        cov = np.cov(X_transposed) 
        eigenvectors, eigenvalues = np.linalg.eig(cov) #unsorted
        # sort eigenvalues and corresponding eigenvectors by eigenvalue in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        sorted_ev = eigenvectors[idxs]
        sorted_evals = eigenvalues[idxs] # if one needs
        self.components = sorted_ev[:self.dims]

    def transform(self, X):
        X = X - self.mean
        return np.dot(self.components, X.T)