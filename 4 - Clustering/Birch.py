#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:27:15 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Création de l'instance de l'algorithme Birch
birch = Birch(threshold=0.5, n_clusters=4)

# Ajustement du modèle aux données
birch.fit(X)

# Récupération des étiquettes de cluster
labels = birch.labels_

# Affichage des résultats
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolors='k')
plt.title('Birch Clustering')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.grid(True)
plt.show()
