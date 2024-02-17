#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:26:22 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import OPTICS

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Création de l'instance de l'algorithme OPTICS
optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)

# Ajustement du modèle aux données
optics.fit(X)

# Extraction des étiquettes de cluster et de la distance de reachability
labels = optics.labels_
reachability = optics.reachability_

# Affichage des résultats
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolors='k')
plt.title('OPTICS Clustering')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.grid(True)
plt.show()
