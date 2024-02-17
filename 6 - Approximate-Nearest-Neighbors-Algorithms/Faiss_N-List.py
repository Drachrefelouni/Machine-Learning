#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:14:43 2023

@author: achref
"""

import numpy as np
import faiss

# Génération de données d'exemple
data = np.random.rand(100, 10).astype('float32')  # 100 points dans un espace de 10 dimensions

# Paramètres FAISS
d = data.shape[1]  # Dimension des données
nlist = 5  # Nombre de listes dans l'index

# Construction de l'index N-List avec FAISS
quantizer = faiss.IndexFlatL2(d)  # L'index quantizer peut être un index plat
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
index.train(data)
index.add(data)

# Requête pour trouver le voisin le plus proche pour un point de requête
query_point = np.random.rand(1, 10).astype('float32')
k = 1  # Nombre de voisins à rechercher
distances, indices = index.search(query_point, k)

print("Index du voisin le plus proche :", indices[0][0])
print("Distance au voisin le plus proche :", distances[0][0])
