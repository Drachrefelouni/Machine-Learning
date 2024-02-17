#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:15:14 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt

# Génération de données aléatoires pour l'exemple
np.random.seed(0)
X = np.random.randn(100, 2)

# Définition de la taille de la grille et de la taille des cellules de grille
grid_size = 10
cell_size = 1.0

# Calcul des coordonnées des cellules de grille
grid_coords = np.arange(0, grid_size * cell_size, cell_size)

# Initialisation de la grille avec des compteurs de densité nuls
density_grid = np.zeros((grid_size, grid_size))

# Compter le nombre de points dans chaque cellule de la grille
for point in X:
    x_idx = np.searchsorted(grid_coords, point[0])
    y_idx = np.searchsorted(grid_coords, point[1])
    if x_idx < grid_size and y_idx < grid_size:
        density_grid[x_idx, y_idx] += 1

# Détermination du cluster de chaque point de données en fonction de sa cellule de grille
def assign_cluster(point, grid_coords):
    x_idx = np.searchsorted(grid_coords, point[0])
    y_idx = np.searchsorted(grid_coords, point[1])
    return (x_idx, y_idx)

# Création d'un dictionnaire pour regrouper les points de données par cluster
cluster_points = {}
for i, point in enumerate(X):
    cluster = assign_cluster(point, grid_coords)
    if cluster not in cluster_points:
        cluster_points[cluster] = []
    cluster_points[cluster].append(point)

# Affichage des points de données pour chaque cluster
plt.figure(figsize=(8, 6))
for cluster, points in cluster_points.items():
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster}', alpha=0.5)

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Points for each Cluster (STING)')
plt.legend()
plt.grid(True)
plt.show()

