#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:54:01 2023

@author: achref
"""
#pip install pykdtree
import numpy as np
from pykdtree.kdtree import KDTree
from sklearn.cluster import KMeans

def product_quantization(X, m, k):
    # X : Matrice de données (n_samples, n_features)
    # m : Nombre de sous-espaces
    # k : Nombre de clusters par sous-espace
    
    n, d = X.shape

    # Diviser l'espace en m sous-espaces
    X_subspaces = np.array_split(X, m, axis=1)

    # Initialiser les centroides pour chaque sous-espace à l'aide de k-means
    centroids = []
    for subspace in X_subspaces:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(subspace)
        centroids.append(kmeans.cluster_centers_)

    # Quantifier chaque vecteur dans chaque sous-espace
    quantized_data = []
    for i in range(m):
        tree = KDTree(centroids[i])
        _, quantization_indices = tree.query(X_subspaces[i], k=1)
        quantized_data.append(quantization_indices.flatten())

    # Concaténer les indices de quantification pour obtenir le vecteur PQ final
    pq_vector = np.vstack(quantized_data).T

    return pq_vector

# Génération de données d'exemple
data = np.random.rand(100, 10)  # 100 points dans un espace de 10 dimensions

# Utilisation de l'algorithme de Product Quantization
m = 2  # Nombre de sous-espaces
k = 5  # Nombre de clusters par sous-espace
pq_result = product_quantization(data, m, k)

print("Résultat de la Product Quantization :")
print(pq_result)
