import numpy as np
from matplotlib import pyplot as plt


x = np.random.randint(5, size=(3,3))


def getDistance(x1, x2):
    return np.sqrt(np.dot(x1-x2, x1-x2))


class KMeans:
    def __init__(self, k=3, epochs = 100, plot_steps = False ):
        self.K = k
        self.epochs = epochs
        self.centroids = []
        self.clusters = [[] for _ in range(self.K)]
        self.plot_steps = plot_steps

    def predict(self,X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Selecting K centroids from X initial samples
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.epochs):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels


    def _create_clusters(self, centroids):
        # assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            print(f"")
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def _closest_centroid(self, sample, centroids):
        distances = [getDistance(sample, c) for c in centroids]
        cluster_idx = np.argmin(distances)
        return cluster_idx
    
    def _is_converged(self, centroids_old, centroids):
        distances = [getDistance(centroids_old[i], centroids[i]) for i in range(self.K)]
        sum_dist = np.sum(distances)
        return sum_dist == 0



    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()