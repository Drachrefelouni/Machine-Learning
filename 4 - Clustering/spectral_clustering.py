#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:02:33 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

# Génération de données aléatoires pour l'exemple (des données en forme de lune)
X, _ = make_moons(n_samples=200, noise=0.1, random_state=0)

# Création de l'instance de l'algorithme de clustering spectral avec 2 clusters
spectral_clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')

# Adapter le modèle aux données et obtenir les étiquettes de cluster
y_spectral = spectral_clustering.fit_predict(X)

# Affichage des points de données avec leurs clusters attribués par clustering spectral
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_spectral, cmap='viridis', s=50, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Spectral Clustering')
plt.show()
