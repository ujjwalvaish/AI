import numpy as np
from sklearn import datasets
from PCA import PCA
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data, iris.target
print(f"Shape of X = {X.shape}")

pca = PCA()
pca.fit(X)
X_projected = pca.transform(X) # dim * n_samples

x1 = X_projected[:1,]
x2 = X_projected[1:,]

plt.scatter(
    x1,x2,c=y
)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.colorbar()
print(f"Shape of X projected = {X_projected.shape}")
plt.show()
# plt(pca_tx)

