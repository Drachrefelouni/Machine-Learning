#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:57:40 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Estimation de la largeur de bande (bandwidth) pour Mean Shift
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# Création de l'instance de l'algorithme Mean Shift
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# Adapter le modèle aux données et obtenir les étiquettes de cluster
y_meanshift = meanshift.fit_predict(X)

# Affichage des points de données avec leurs clusters attribués par Mean Shift
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_meanshift, cmap='viridis', s=50, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Mean Shift Clustering')
plt.show()
