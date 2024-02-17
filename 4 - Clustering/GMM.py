#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:58:41 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Création de l'instance du modèle de mélange gaussien avec 4 composants
gmm = GaussianMixture(n_components=4)

# Adapter le modèle aux données
gmm.fit(X)

# Prédiction des clusters pour chaque exemple dans les données
y_gmm = gmm.predict(X)

# Affichage des points de données avec leurs clusters attribués par GMM
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis', s=50, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Mixture Models Clustering')
plt.show()
