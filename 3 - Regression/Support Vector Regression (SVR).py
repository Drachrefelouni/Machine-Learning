# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Charger l'ensemble de données Iris
iris = load_iris()
X = iris.data  # Les caractéristiques
y = iris.target  # Les étiquettes

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Réduction de dimensionnalité avec PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Création du modèle SVR
svr = SVR(kernel='linear')

# Entraînement du modèle
svr.fit(X_train_pca, y_train)

# Prédiction sur l'ensemble de test
y_pred = svr.predict(X_test_pca)

# Calcul de l'erreur de prédiction
mse = mean_squared_error(y_test, y_pred)
print("Erreur quadratique moyenne (MSE) :", mse)

# Visualisation des résultats
plt.figure(figsize=(8, 6))

# Données réelles
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.Set1, edgecolor='k', s=40, label='Vraies étiquettes')

# Prédictions
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap=plt.cm.Paired, edgecolor='k', s=80, marker='s', label='Prédictions')

plt.xlabel('Premier composant principal')
plt.ylabel('Deuxième composant principal')
plt.title('Prédictions SVR avec PCA (MSE={:.2f})'.format(mse))
plt.legend()
plt.show()
