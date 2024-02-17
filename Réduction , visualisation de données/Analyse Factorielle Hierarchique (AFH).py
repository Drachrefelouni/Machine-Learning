#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:48:49 2024

@author: achref
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Chargement des données Iris
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Construction de la matrice de liaison avec la méthode 'ward'
Z = linkage(X_scaled, method='ward')

# Tracé du dendrogramme
plt.figure(figsize=(10, 7))
plt.title("Dendrogramme de l'Analyse Factorielle Hiérarchique (AFH)")
dendrogram(Z, labels=iris.target, orientation='right', leaf_rotation=90)
plt.xlabel('Échantillons')
plt.ylabel('Distance')
plt.axhline(y=7, color='r', linestyle='--')
plt.show()
