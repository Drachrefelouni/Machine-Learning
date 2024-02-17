#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:55:50 2024

@author: achref
"""

# Import des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# Chargement des données Iris
iris = load_iris()
X = iris.data[:, :2]  # Nous ne prenons que les deux premières fonctionnalités pour une visualisation plus facile
Y = iris.target

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialisation du modèle de régression logistique
model = LogisticRegression()

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, Y_train)

# Prédiction sur l'ensemble de test
Y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(Y_test, Y_pred)
print("Précision du modèle :", accuracy)

# Matrice de confusion
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Matrice de confusion :\n", conf_matrix)

# Plot
# Création d'une grille pour tracer la frontière de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.02  # pas de la grille
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Prédiction pour chaque point de la grille
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Affichage de la frontière de décision
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Affichage des points d'entraînement
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Paired, edgecolors='k')
plt.xlabel('Longueur des sépales')
plt.ylabel('Largeur des sépales')
plt.title('Régression logistique sur les données Iris')
plt.show()
