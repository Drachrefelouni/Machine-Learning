#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:43:45 2024

@author: achref
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import dendrogram

# Générer des données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# Paramètres pour CURE
k = 10  # Nombre de représentants à sélectionner
r = 0.1  # Fraction de points à fusionner à chaque itération

# Créer un graphe de k-plus proches voisins
connectivity = kneighbors_graph(X, n_neighbors=k, include_self=False)

# Appliquer l'algorithme de clustering agglomératif avec CURE
model = AgglomerativeClustering(n_clusters=None, linkage='single', connectivity=connectivity, distance_threshold=0)
model.fit(X)

# Obtenir les labels de cluster et les distances
labels = model.labels_
distances = model.distances_

# Visualisation du dendrogramme
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Un nœud terminal, donc compter un seul
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plt.figure(figsize=(10, 5))
plt.title('Dendrogramme CURE')
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Points d'observation")
plt.ylabel('Distance')
plt.show()

# Afficher les clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('CURE Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()
