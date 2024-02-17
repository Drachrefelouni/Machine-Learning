#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:04:16 2023

@author: achref
"""

#pip install annoy
from annoy import AnnoyIndex
import numpy as np

def build_annoy_index(data, num_trees):
    # Créer un index Annoy
    dim = data.shape[1]
    annoy_index = AnnoyIndex(dim, metric='euclidean')

    # Ajouter des vecteurs à l'index
    for i, vector in enumerate(data):
        annoy_index.add_item(i, vector.tolist())

    # Construire l'index avec le nombre d'arbres spécifié
    annoy_index.build(num_trees)

    return annoy_index

def annoy_search(query_vector, annoy_index, num_neighbors):
    # Rechercher les k voisins les plus proches
    nearest_neighbors, distances = annoy_index.get_nns_by_vector(query_vector.tolist(), num_neighbors, include_distances=True)

    return nearest_neighbors, distances

# Génération de données d'exemple
data = np.random.rand(100, 10)  # 100 points dans un espace de 10 dimensions

# Construction de l'index Annoy
num_trees = 10  # Nombre d'arbres dans l'index
annoy_index = build_annoy_index(data, num_trees)

# Point de requête
query_vector = np.random.rand(10)

# Recherche des k voisins les plus proches à l'aide d'Annoy
num_neighbors = 5
nearest_neighbors, distances = annoy_search(query_vector, annoy_index, num_neighbors)

print("Index des k voisins les plus proches :", nearest_neighbors)
print("Distances correspondantes :", distances)
