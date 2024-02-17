#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:12:30 2024

@author: achref
"""

# Import des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns

# Chargement des données Iris
iris = load_iris()
X = iris.data
Y = iris.target

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Initialisation du modèle ANN
model = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', random_state=42)

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