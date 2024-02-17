#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:36:55 2024

@author: achref
"""

# Import des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris

# Chargement des données Iris
iris = load_iris()
X = iris.data[:, :1]  # Sélection de la première caractéristique pour l'exemple
Y = iris.target

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialisation du modèle d'arbre de décision pour la régression
model = DecisionTreeRegressor()

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
plt.scatter(X_test, Y_test, color='black')
plt.plot(np.sort(X_test, axis=0), np.sort(Y_pred, axis=0), color='blue', linewidth=3)
plt.xlabel('Caractéristique')
plt.ylabel('Classe')
plt.title('Arbre de décision pour la régression sur les données Iris')
plt.show()
