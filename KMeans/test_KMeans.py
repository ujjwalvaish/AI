from sklearn.datasets import make_blobs
from KMeans import KMeans

X, y = make_blobs(n_samples=500, n_features=2, centers=5, shuffle=True, random_state=40)

model = KMeans(k=5)
predictions = model.predict(X)

model.plot()
