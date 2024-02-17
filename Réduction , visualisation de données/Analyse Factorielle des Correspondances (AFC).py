#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:20:55 2024

@author: achref
"""

import pandas as pd
from prince import CA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Charger le jeu de données Iris
iris = load_iris()
X = iris.data
y = iris.target

# Convertir les données en DataFrame pandas
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y

# Appliquer l'Analyse Factorielle des Correspondances (AFC)
ca = CA(n_components=2)
ca.fit(iris_df.drop('target', axis=1))

# Obtenir les coordonnées des lignes dans l'espace des composantes principales
coordinates = ca.row_coordinates(iris_df.drop('target', axis=1))

# Visualiser les résultats
plt.figure(figsize=(8, 6))
plt.scatter(coordinates.iloc[:, 0], coordinates.iloc[:, 1], c=y, cmap='viridis')
plt.title('Analyse Factorielle des Correspondances (AFC) - Iris Dataset')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.colorbar(label='Classe')
plt.grid(True)
plt.show()
