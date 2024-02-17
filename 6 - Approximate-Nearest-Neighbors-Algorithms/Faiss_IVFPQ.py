#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:13:10 2023

@author: achref
"""
#pip install faiss
import numpy as np
import faiss

# Génération de données d'exemple
data = np.random.rand(100, 10).astype('float32')  # 100 points dans un espace de 10 dimensions

# Paramètres FAISS
d = data.shape[1]  # Dimension des données
nlist = 5  # Nombre de listes dans l'index
m = 8  # Paramètre m pour IVFPQ (nombre de quantizers)
nbits = 8  # Paramètre nbits pour IVFPQ (nombre de bits par quantizer)

# Construction de l'index IVFPQ avec FAISS
quantizer = faiss.IndexFlatL2(d)  # L'index quantizer peut être un index plat
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.train(data)
index.add(data)

# Requête pour trouver le voisin le plus proche pour un point de requête
query_point = np.random.rand(1, 10).astype('float32')
k = 1  # Nombre de voisins à rechercher
distances, indices = index.search(query_point, k)

print("Index du voisin le plus proche :", indices[0][0])
print("Distance au voisin le plus proche :", distances[0][0])
