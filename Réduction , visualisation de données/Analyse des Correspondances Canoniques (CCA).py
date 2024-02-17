#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:32:05 2024

@author: achref
"""

from sklearn.cross_decomposition import CCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Charger le jeu de données Iris
iris = load_iris()
X = iris.data[:, :2]  # Sélectionner seulement les deux premières caractéristiques pour cet exemple
Y = iris.data[:, 2:]  # Sélectionner les deux dernières caractéristiques pour cet exemple

# Appliquer l'Analyse des Correspondances Canoniques (CCA)
cca = CCA(n_components=2)
cca.fit(X, Y)

# Transformer les données
X_c, Y_c = cca.transform(X, Y)

# Visualiser les résultats
plt.figure(figsize=(8, 6))
plt.scatter(X_c[:, 0], X_c[:, 1], c=iris.target, cmap='viridis', label='X')
plt.scatter(Y_c[:, 0], Y_c[:, 1], c=iris.target, cmap='viridis', marker='s', label='Y')
plt.title('Analyse des Correspondances Canoniques (CCA)')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.legend()
plt.grid(True)
plt.show()
