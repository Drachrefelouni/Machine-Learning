#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:06:38 2023

@author: achref
"""

#pip install mrpt
from mrpt import MRPTIndex
import numpy as np

# Génération de données d'exemple
data = np.random.rand(100, 10)  # 100 points dans un espace de 10 dimensions

# Construction de l'index MRPT
num_trees = 5  # Nombre d'arbres aléatoires
mrpt_index = MRPTIndex(data, trees=num_trees)

# Requête pour trouver le voisin le plus proche d'un point de requête
query_point = np.random.rand(1, 10)
nearest_neighbor_index, nearest_neighbor_distance = mrpt_index.knn_query(query_point, k=1)

print("Index du voisin le plus proche :", nearest_neighbor_index)
print("Distance au voisin le plus proche :", nearest_neighbor_distance)
