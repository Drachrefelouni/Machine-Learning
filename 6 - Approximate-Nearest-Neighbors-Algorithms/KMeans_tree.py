#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:02:32 2023

@author: achref
"""
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist

def kmeans_tree_search(query_point, dataset, num_clusters):
    # Appliquer K-Means pour diviser l'espace des données en clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(dataset)
    cluster_assignments = kmeans.predict(query_point)

    # Extraire les points du cluster correspondant au point de requête
    cluster_points = dataset[cluster_assignments[0] == kmeans.labels_]

    # Utiliser la recherche brute-force dans le cluster
    nearest_index, nearest_distance = brute_force_search(query_point, cluster_points)

    # Ajuster l'indice pour prendre en compte le cluster
    nearest_index_global = np.where(cluster_assignments[0] == kmeans.labels_)[0][nearest_index]

    return nearest_index_global, nearest_distance

# Fonction brute-force pour la recherche à l'intérieur des clusters
def brute_force_search(query_point, dataset):
    distances = cdist(query_point, dataset, metric='euclidean')
    nearest_neighbor_index = np.argmin(distances)
    nearest_neighbor_distance = distances[0, nearest_neighbor_index]

    return nearest_neighbor_index, nearest_neighbor_distance

# Génération de données d'exemple
query_point = np.random.rand(1, 10)  # Point de requête dans un espace de 10 dimensions
dataset = np.random.rand(100, 10)  # 100 points dans un espace de 10 dimensions
num_clusters = 5  # Nombre de clusters

# Recherche du plus proche voisin en utilisant K-Means Tree
nearest_index, nearest_distance = kmeans_tree_search(query_point, dataset, num_clusters)

print("Index du plus proche voisin :", nearest_index)
print("Distance au plus proche voisin :", nearest_distance)
