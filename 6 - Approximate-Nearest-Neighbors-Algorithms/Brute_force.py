#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:01:31 2023

@author: achref
"""

import numpy as np
from scipy.spatial.distance import cdist

def brute_force_search(query_point, dataset):
    distances = cdist(query_point, dataset, metric='euclidean')
    nearest_neighbor_index = np.argmin(distances)
    nearest_neighbor_distance = distances[0, nearest_neighbor_index]

    return nearest_neighbor_index, nearest_neighbor_distance

# Génération de données d'exemple
query_point = np.random.rand(1, 10)  # Point de requête dans un espace de 10 dimensions
dataset = np.random.rand(100, 10)  # 100 points dans un espace de 10 dimensions

# Recherche du plus proche voisin en utilisant la recherche brute-force
nearest_index, nearest_distance = brute_force_search(query_point, dataset)

print("Index du plus proche voisin :", nearest_index)
print("Distance au plus proche voisin :", nearest_distance)
