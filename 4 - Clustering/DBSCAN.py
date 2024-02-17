#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:56:59 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Génération de données aléatoires pour l'exemple (des données en forme de lune)
X, _ = make_moons(n_samples=200, noise=0.1, random_state=0)

# Création de l'instance de l'algorithme DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)

# Adapter le modèle aux données et obtenir les étiquettes de cluster
y_dbscan = dbscan.fit_predict(X)

# Affichage des points de données avec leurs clusters attribués par DBSCAN
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis', s=50, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.show()
