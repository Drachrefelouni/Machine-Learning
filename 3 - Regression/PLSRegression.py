#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:35:16 2024

@author: achref
"""

# Import des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris

# Chargement des données Iris
iris = load_iris()
X = iris.data[:, :2]  # Sélection des deux premières caractéristiques pour l'exemple
Y = iris.target

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialisation du modèle PLS
n_components = 2  # Nombre de composants PLS
model = PLSRegression(n_components=n_components)

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, Y_train)

# Prédiction sur l'ensemble de test
Y_pred = model.predict(X_test)

# Calcul des métriques d'évaluation
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Squared Error:", mse)
print("Coefficient de détermination (R²):", r2)

# Plot des données et de la courbe de régression
plt.scatter(Y_test, Y_pred, color='blue')
plt.xlabel('Vraies valeurs')
plt.ylabel('Prédictions')
plt.title('Régression PLS sur les données Iris')
plt.show()
