#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:21:20 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation
from itertools import cycle

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Création de l'instance de l'algorithme Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X)

# Récupération des centres de cluster
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

# Nombre de clusters obtenus
n_clusters_ = len(cluster_centers_indices)

# Affichage des résultats
plt.figure(figsize=(8, 6))
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Affinity Propagation Clustering')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.grid(True)
plt.show()
