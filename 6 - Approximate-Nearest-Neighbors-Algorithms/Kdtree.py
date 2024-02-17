#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:49:01 2023

@author: achref
"""

from sklearn.neighbors import KDTree
import numpy as np

# Génération de données d'exemple
data = np.random.rand(100, 2)  # 100 points dans un espace bidimensionnel

# Construction de l'arbre KD
kdtree = KDTree(data, leaf_size=30)

# Requête pour trouver les k voisins les plus proches d'un point de requête
query_point = np.random.rand(1, 2)
k_neighbors = 5
distances, indices = kdtree.query(query_point, k=k_neighbors)

print("Index des k voisins les plus proches :", indices)
print("Distances correspondantes :", distances)
