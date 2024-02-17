#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:09:23 2023

@author: achref
"""

import falconn
import numpy as np

# Génération de données d'exemple
data = np.random.rand(100, 10).astype(np.float32)  # 100 points dans un espace de 10 dimensions

# Configuration des paramètres FALCONN
parameters = falconn.LSHConstructionParameters()
parameters.dimension = data.shape[1]
parameters.lsh_family = falconn.LSHFamily.CrossPolytope
parameters.distance_function = falconn.DistanceFunction.EuclideanSquared
parameters.k = 1  # Nombre de voisins les plus proches à rechercher
parameters.l = 50  # Nombre de tables de hachage

# Construction de la table de hachage FALCONN
table = falconn.LSHIndex(parameters)
table.setup(data)

# Recherche du voisin le plus proche pour un point de requête
query_point = np.random.rand(1, 10).astype(np.float32)
result = table.find_nearest_neighbors(query_point, 1)

print("Index du voisin le plus proche :", result[0])
print("Distance au voisin le plus proche :", result[1])
