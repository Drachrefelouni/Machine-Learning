#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:51:43 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Création de l'instance de l'algorithme K-means avec 4 clusters
kmeans = KMeans(n_clusters=4)

# Adapter le modèle aux données
kmeans.fit(X)

# Prédire les clusters pour chaque exemple dans les données
y_kmeans = kmeans.predict(X)

# Obtenir les coordonnées des centres de cluster
centers = kmeans.cluster_centers_

# Tracer les points de données et les centres de cluster
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering with K-means')
plt.show()
