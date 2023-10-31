import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, tolerance=0.001, max_iter = 500) -> None:
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance

    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2, axis=0)
    
    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iterations):
            self.classes = {}
            for j in range(self.k):
                self.classes[j] = []
            
            for point in data:
                distances = []
                for index in self.centroids:
                    distances.append(self.euclidean_distance(point, self.centroids[index]))
                cluster_index = distances.index(min(distances))
                self.classes[cluster_index].append(point)

            previous = dict(self.centroids)
            for cluster_index in self.classes:
                self.centroids[cluster_index] = np.average(self.classes[cluster_index], axis=0)
            
            isOptimal = True
            for centroid in self.centroids:
                original_centroid  = previous[centroid]
                curr_centroid = self.centroids[centroid]
                if np.sum((curr_centroid - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                break

if __name__ == '__main__':
    #generate dummy cluster datasets
    # Set three centers, the model should predict similar results
    center_1 = np.array([1,1])
    center_2 = np.array([5,5])
    center_3 = np.array([8,1])

    # Generate random data and center it to the three centers
    cluster_1 = np.random.randn(100, 2) + center_1
    cluster_2 = np.random.randn(100,2) + center_2
    cluster_3 = np.random.randn(100,2) + center_3

    data = np.concatenate((cluster_1, cluster_2, cluster_3), axis = 0)

    # Here we have created 3 groups of data of two dimension with different centre. 
    # We have defined the value of k as 3. Now lets fit the model created from scratch.
    k = 5
    k_means = KMeans(k=k)
    k_means.fit(data)
    
    
    # Plotting starts here
    colors = 10*["r", "g", "c", "b", "k"]

    for centroid in k_means.centroids:
        plt.scatter(k_means.centroids[centroid][0], k_means.centroids[centroid][1], s = 130, marker = "x")

    for cluster_index in k_means.classes:
        color = colors[cluster_index]
        for features in k_means.classes[cluster_index]:
            plt.scatter(features[0], features[1], color = color,s = 30)

    plt.show() 

