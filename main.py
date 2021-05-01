# Source: https://blog.quantinsti.com/hierarchical-clustering-python/#agglomerative-hierarchical-clustering
#Github:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering




# Reading values from the input file
dataset = pd.read_csv('input.csv',delimiter=";", header = None, skiprows=2)
X = np.array(dataset)

# Plot the original data
plt.figure('Original Data')
plt.title('Original Data')
plt.ylabel('y')
plt.xlabel('x')
plt.scatter(X[:, 0], X[:, 1], color='blue')
plt.show()

# Fit the model
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
label = cluster.fit_predict(X)

# Plot the clusters
plt.figure('Hirachical Clustering')
plt.scatter(X[label == 0, 0], X[label == 0, 1], s=100, marker='o', color='red', label='Cluster 1')
plt.scatter(X[label == 1, 0], X[label == 1, 1], s=100, marker='o', color='blue', label='Cluster 2')
plt.scatter(X[label == 2, 0], X[label == 2, 1], s=100, marker='o', color='green', label='Cluster 3')
plt.title('Hirachical Clustering')
plt.legend(loc='upper left')
plt.ylabel('y')
plt.xlabel('x')
plt.show()

# Plot the Dendrogram
plt.figure('Dendrogram')
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Euclidean distances')
plt.show()
