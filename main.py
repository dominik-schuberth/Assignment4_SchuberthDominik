import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# https://blog.quantinsti.com/hierarchical-clustering-python/#agglomerative-hierarchical-clustering
# https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019

dataset = pd.read_csv('input.csv',delimiter=";", header = None, skiprows=2)
X = np.array(dataset)

plt.figure('Data')
plt.ylabel('y')
plt.xlabel('x')
plt.scatter(X[:, 0], X[:, 1], color='blue')
plt.show()

print("alt", X)

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit(X) # the fit methode tells for each data point to which cluster the point belongs to
print(X)
labels = cluster.labels_
print(labels)

plt.scatter(X[labels == 0, 0], X[labels == 0, 1], marker='o', color='red')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], marker='o', color='blue')
plt.scatter(X[labels == 2, 0], X[labels == 2, 1], marker='o', color='green')
# plt.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
# plt.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='orange')

plt.ylabel('y')
plt.xlabel('x')
plt.show()

dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title('Dendrogram')
#plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()