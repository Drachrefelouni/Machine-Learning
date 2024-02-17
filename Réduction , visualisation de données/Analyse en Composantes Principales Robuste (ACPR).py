#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:35:22 2024

@author: achref
"""
#pip install factor-analyzer
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import RobustScaler
import numpy as np

# Charger les données Iris
iris = load_iris()
X = iris.data

# Standardiser les données en utilisant RobustScaler
robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X)

# Appliquer l'ACP basée sur la médiane (ACPR)
pca = PCA(n_components=2)
pca.fit(X_scaled)

# Afficher les composantes principales
print("Composantes principales :")
print(pca.components_)

# Afficher les proportions de variance expliquée
print("Proportions de variance expliquée :")
print(pca.explained_variance_ratio_)

# Transformer les données
X_transformed = pca.transform(X_scaled)
import matplotlib.pyplot as plt

# Tracer les données projetées sur les deux premières composantes principales
plt.figure(figsize=(8, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=iris.target, cmap='viridis')
plt.title('Projection des données sur les deux premières composantes principales (ACPR)')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.colorbar(label='Classe')
plt.grid(True)
plt.show()
