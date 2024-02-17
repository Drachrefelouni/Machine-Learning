#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:02:27 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Charger les données d'exemple (iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Effectuer l'ACP avec 2 composantes principales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Afficher les données projetées sur les deux premières composantes principales
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
plt.title('ACP - Projection des données Iris')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.colorbar(label='Classe')
plt.grid(True)
plt.show()
