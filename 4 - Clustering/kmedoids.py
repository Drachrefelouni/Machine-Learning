#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:03:42 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Création de l'instance de l'algorithme des k-medoids avec 4 clusters
kmedoids = KMedoids(n_clusters=4)

# Adapter le modèle aux données
kmedoids.fit(X)

# Obtention des indices des médoides et des étiquettes de cluster
medoid_indices = kmedoids.medoid_indices_
labels = kmedoids.labels_

# Affichage des points de données avec leurs clusters attribués par k-medoids
plt.figure(figsize=(8, 6))
for i in range(4):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
plt.scatter(X[medoid_indices, 0], X[medoid_indices, 1], c='black', marker='x', label='Medoids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Medoids Clustering')
plt.legend()
plt.show()

