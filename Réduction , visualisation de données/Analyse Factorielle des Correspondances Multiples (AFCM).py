#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:02:26 2024

@author: achref
"""
import pandas as pd
from prince import MCA
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Charger le jeu de données Iris
iris = load_iris()
X = iris.data
y = iris.target

# Convertir les données en DataFrame pandas
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y

# Séparer les variables catégorielles et continues
variables_catégorielles = ['target']
variables_continues = iris.feature_names

# Appliquer l'Analyse Factorielle des Correspondances Multiples (AFCM)
mca = MCA(n_components=2)
mca.fit(iris_df[variables_catégorielles])

pca = PCA(n_components=2)
pca.fit(iris_df[variables_continues])

# Visualiser les résultats
plt.figure(figsize=(8, 6))
plt.scatter(mca.row_coordinates(iris_df[variables_catégorielles]).iloc[:, 0], 
            mca.row_coordinates(iris_df[variables_catégorielles]).iloc[:, 1], 
            c=y, cmap='viridis', label='Variables catégorielles')
plt.scatter(pca.transform(iris_df[variables_continues])[:, 0], 
            pca.transform(iris_df[variables_continues])[:, 1], 
            c=y, cmap='viridis', marker='s', label='Variables continues')
plt.title('Analyse Factorielle des Correspondances Multiples (AFCM) - Iris Dataset')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.legend()
plt.grid(True)
plt.show()

