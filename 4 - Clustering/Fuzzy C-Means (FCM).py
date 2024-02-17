#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:41:33 2024

@author: achref
"""
#pip install scikit-fuzzy

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Générer des données aléatoires pour l'exemple
np.random.seed(42)
n_points = 1000
n_features = 2
data = np.random.uniform(0, 10, (n_points, n_features))

# Paramètres pour FCM
num_clusters = 3
m = 2  # Fuzziness coefficient (usually set to 2 for FCM)

# Appliquer FCM
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data.T, num_clusters, m, error=0.005, maxiter=1000)

# Obtenir les clusters
clusters = np.argmax(u, axis=0)

# Visualiser les clusters
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.scatter(cntr[0], cntr[1], marker='*', s=300, c='red', label='Centroids')
plt.title('FCM Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.legend()
plt.grid(True)
plt.show()
