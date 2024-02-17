#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:07:50 2024

@author: achref
"""
import pandas as pd
from prince import MCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Charger le jeu de données Iris
iris = load_iris()

# Convertir les données en DataFrame pandas
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Discrétiser les données en données catégorielles
for feature in iris_df.columns:
    iris_df[feature] = pd.cut(iris_df[feature], bins=3, labels=['Faible', 'Moyen', 'Élevé'])

# Appliquer l'ACM
mca = MCA()
mca.fit(iris_df)

# Extraire les coordonnées des lignes (observations)
coordinates = mca.row_coordinates(iris_df)

# Afficher les résultats
plt.figure(figsize=(8, 6))
plt.scatter(coordinates.iloc[:, 0], coordinates.iloc[:, 1], c=iris.target, cmap='viridis')
plt.title('Analyse des Correspondances Multiples (ACM) - Iris Dataset')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.colorbar(label='Classe')
plt.grid(True)
plt.show()
