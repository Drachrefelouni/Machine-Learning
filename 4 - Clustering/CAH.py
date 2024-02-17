
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:51:43 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Génération de données aléatoires pour l'exemple
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Création de l'instance de l'algorithme de clustering hiérarchique avec 4 clusters
agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0)

# Adapter le modèle aux données et obtenir les distances et les indices des clusters fusionnés
y_agg = agg_clustering.fit_predict(X)

# Calcul de la matrice de liaison
Z = linkage(X, method='ward')  # Méthode de liaison pour construire la matrice de liaison

# Affichage du dendrogramme
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
dendrogram(Z, labels=agg_clustering.labels_)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
