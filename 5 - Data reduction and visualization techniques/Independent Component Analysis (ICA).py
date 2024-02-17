#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:15:50 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import FastICA

# Charger le jeu de données Iris
iris = load_iris()
X = iris.data
y = iris.target

# Appliquer l'ACI pour réduire la dimensionnalité en 2 composantes
ica = FastICA(n_components=2, random_state=42)
X_ica = ica.fit_transform(X)

# Visualiser les données dans l'espace de composantes indépendantes
plt.figure(figsize=(8, 6))
plt.scatter(X_ica[:, 0], X_ica[:, 1], c=y, cmap='viridis')
plt.title('Analyse en Composantes Indépendantes (ACI) - Iris Dataset')
plt.xlabel('Composante indépendante 1')
plt.ylabel('Composante indépendante 2')
plt.colorbar(label='Classe')
plt.grid(True)
plt.show()
