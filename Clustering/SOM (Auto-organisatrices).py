#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:09:45 2024

@author: achref
"""
#pip install MiniSom

import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import make_blobs

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Normalisation des données
X_normalized = (X - X.mean()) / X.std()

# Création de la carte auto-organisatrice
som = MiniSom(2, 2, X_normalized.shape[1], sigma=1.0, learning_rate=0.5)

# Initialisation aléatoire des poids
som.random_weights_init(X_normalized)

# Entraînement de la carte auto-organisatrice
som.train_random(X_normalized, 100)

# Trouver les clusters pour chaque point de données
winners = np.array([som.winner(x) for x in X_normalized])

# Créer un dictionnaire pour regrouper les points par cluster
cluster_points = {}
for i, winner in enumerate(winners):
    cluster = tuple(winner)
    if cluster not in cluster_points:
        cluster_points[cluster] = []
    cluster_points[cluster].append(X[i])

# Affichage des points de données pour chaque cluster
plt.figure(figsize=(10, 10))
for cluster, points in cluster_points.items():
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster}', alpha=0.5)

plt.title('Points for each Cluster (SOM)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()

