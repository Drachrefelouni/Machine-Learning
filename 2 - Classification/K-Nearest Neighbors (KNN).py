#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:08:16 2024

@author: achref
"""

# Import des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns

# Chargement des données Iris
iris = load_iris()
X = iris.data
Y = iris.target

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialisation du modèle des k plus proches voisins (KNN)
model = KNeighborsClassifier(n_neighbors=3)

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

# Plot de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs réelles')
plt.title('Matrice de confusion de la forêt aléatoire')
plt.show()