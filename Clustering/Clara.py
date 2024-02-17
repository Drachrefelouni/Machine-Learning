#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:31:04 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clara(data, num_samples, num_clusters):
    best_silhouette_score = -1
    best_centers = None
    
    for _ in range(num_samples):
        sample_indices = np.random.choice(len(data), size=num_samples, replace=False)
        sample_data = data[sample_indices]
        kmeans = KMeans(n_clusters=num_clusters)
        labels = kmeans.fit_predict(sample_data)
        silhouette_avg = silhouette_score(sample_data, labels)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_centers = kmeans.cluster_centers_
    
    return best_centers

# Générer des données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=1.0, random_state=42)

# Paramètres pour CLARA
num_samples = 5
num_clusters = 4

# Appliquer CLARA
centers = clara(X, num_samples, num_clusters)

# Visualisation des clusters
plt.scatter(X[:, 0], X[:, 1], c='lightgray', marker='.', label='Data points')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=300, label='Cluster centers')
plt.title('CLARA Clustering')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
