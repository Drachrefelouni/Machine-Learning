#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:46:16 2024

@author: achref
"""

# Import des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Chargement des données Iris
iris = load_iris()
X = iris.data[:, :2]  # Sélection de deux caractéristiques pour la visualisation
Y = iris.target

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialisation du modèle QDA
model = QuadraticDiscriminantAnalysis()

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, Y_train)

# Prédiction sur l'ensemble de test
Y_pred = model.predict(X_test)

# Calcul de l'exactitude
accuracy = accuracy_score(Y_test, Y_pred)
print("Exactitude du modèle:", accuracy)

# Plot des frontières de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
plt.xlabel('Longueur du sépale')
plt.ylabel('Largeur du sépale')
plt.title('Classification QDA sur les données Iris')
plt.show()
