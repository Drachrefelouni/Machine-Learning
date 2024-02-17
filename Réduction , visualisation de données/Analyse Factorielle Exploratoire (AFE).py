#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:35:08 2024

@author: achref
"""

from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Charger le jeu de données Iris
iris = load_iris()
X = iris.data

# Appliquer l'Analyse Factorielle Exploratoire (AFE)
afe = FactorAnalysis(n_components=2)
X_transformed = afe.fit_transform(X)

# Visualiser les résultats
plt.figure(figsize=(8, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=iris.target, cmap='viridis')
plt.title('Analyse Factorielle Exploratoire (AFE)')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.grid(True)
plt.show()
