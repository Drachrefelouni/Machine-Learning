#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:46:58 2024

@author: achref
"""

from sklearn.datasets import load_iris

# Chargement des données Iris
iris = load_iris()

# Accès aux caractéristiques et aux étiquettes
X = iris.data  # Caractéristiques
Y = iris.target  # Étiquettes

# Affichage des informations sur le jeu de données
print("Forme des caractéristiques:", X.shape)
print("Forme des étiquettes:", Y.shape)
print("Noms des classes:", iris.target_names)
print("Noms des caractéristiques:", iris.feature_names)
