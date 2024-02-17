#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:49:04 2024

@author: achref
"""
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

import numpy as np
from scipy.spatial.distance import cdist

def gaussian_kernel(x, xi, h):
    return np.exp(-np.sum((x - xi) ** 2) / (2 * h ** 2))

def denclue(X, h, eps, min_density):
    n = X.shape[0]
    density = np.zeros(n)
    cluster_labels = -np.ones(n, dtype=int)
    
    for i in range(n):
        density[i] = np.sum(gaussian_kernel(X[i], X, h))
    
    for i in range(n):
        if density[i] > min_density:
            cluster_labels[i] = i
    
    for i in range(n):
        if cluster_labels[i] != -1:
            continue
        for j in range(n):
            if cluster_labels[j] != -1 and cdist(X[i:i+1], X[j:j+1])[0][0] < eps and density[i] > min_density:
                cluster_labels[i] = cluster_labels[j]
                break
    
    return cluster_labels

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer des données aléatoires pour l'exemple
    np.random.seed(0)
    X = np.random.randn(100, 2)
    
    # Appliquer l'algorithme DENCLUE
    cluster_labels = denclue(X, h=0.5, eps=0.1, min_density=0.2)
    
    # Afficher les clusters
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.title('DENCLUE Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()


