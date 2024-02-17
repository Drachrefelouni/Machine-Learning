#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:31:56 2024

@author: achref
"""
# pip install pyclustering
from pyclustering.cluster.clarans import clarans, clarans_observer
from pyclustering.utils import read_sample, distance_metric, type_metric
from pyclustering.samples.definitions import FCPS_SAMPLES

# Charger les données à partir d'un fichier ou utiliser un jeu de données prédéfini
path_to_data = FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS
sample = read_sample(path_to_data)

# Paramètres pour CLARANS
num_clusters = 2
num_local = 2
max_neighbors = 3

# Définir une métrique de distance (euclidienne)
metric = distance_metric(type_metric.EUCLIDEAN)

# Créer une instance de l'objet CLARANS
clarans_instance = clarans(sample, num_clusters, num_local, max_neighbors, metric)

# Observer le processus de clustering (optionnel)
observer = clarans_observer()
clarans_instance.set_observer(observer)

# Exécuter l'algorithme CLARANS
clarans_instance.process()

# Obtenir les clusters résultants
clusters = clarans_instance.get_clusters()

# Afficher les clusters
print("Nombre de clusters trouvés:", len(clusters))
for i, cluster in enumerate(clusters):
    print("Cluster", i+1, ":", cluster)

# Visualiser le processus de clustering (optionnel)
# observer.show_clusters(sample, clarans_instance.get_clusters(), clarans_instance.get_medoids())

