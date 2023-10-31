import numpy as np

def initialize_centroids(points, k):
    """Randomly initialize centroids"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroids(points, centroids):
    """Find the closest centroids for all points"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def recompute_centroids(points, closest, centroids):
    """Recompute centroids"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def kmeans(points, k, max_iters=1000):
    centroids = initialize_centroids(points, k)

    for _ in range(max_iters):
        closest = closest_centroids(points, centroids)
        centroids = recompute_centroids(points, closest, centroids)
    return closest

points = np.random.rand(1000, 2)
k = 5
labels = kmeans(points, k)

import matplotlib.pyplot as plt
for i in range(k):
    plt.scatter(points[labels == i, 0], points[labels == i, 1], label=f"Cluster {i}")

plt.legend()
plt.show()
