#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:45:37 2023

@author: achref
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np

# Génération de données d'exemple
data = np.random.rand(100, 10)  # 100 points dans un espace de 10 dimensions

# Configuration de LSH avec NearestNeighbors
lsh = NearestNeighbors(
    n_neighbors=5,  # Nombre de voisins à rechercher
    algorithm='ball_tree',  # Vous pouvez essayer également 'kd_tree' ou 'brute'
    metric='euclidean'  # La distance utilisée pour la recherche des voisins
)

# Apprentissage de LSH
lsh.fit(data)

# Requête LSH pour trouver des voisins approximatifs d'un point de requête
query_point = np.random.rand(1, 10)
distances, neighbors = lsh.kneighbors(query_point)

print("Index des voisins approximatifs :", neighbors)
print("Distances correspondantes :", distances)