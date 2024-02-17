#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:56:27 2023

@author: achref
"""
#pip install nmslib
import nmslib
import numpy as np

# Génération de données d'exemple
data = np.random.rand(100, 10)  # 100 points dans un espace de 10 dimensions

# Configuration de l'index HNSW
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
index.createIndex({'post': 2}, print_progress=True)

# Requête pour trouver les k voisins les plus proches d'un point de requête
query_point = np.random.rand(1, 10)
k_neighbors = 5
ids, distances = index.knnQuery(query_point, k=k_neighbors)

print("Index des k voisins les plus proches :", ids)
print("Distances correspondantes :", distances)
