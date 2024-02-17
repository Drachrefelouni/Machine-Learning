#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:13:14 2024

@author: achref
"""

# pip install hdbscan

import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.datasets import make_blobs

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Création de l'instance de l'algorithme HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)

# Adapter le modèle aux données et obtenir les étiquettes de cluster
y_hdbscan = clusterer.fit_predict(X)

# Affichage des points de données avec leurs clusters attribués par HDBSCAN
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_hdbscan, cmap='viridis', s=50, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('HDBSCAN Clustering')
plt.show()
