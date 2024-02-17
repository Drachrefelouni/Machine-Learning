#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:52:22 2023

@author: achref
"""
#pip install hyperopt
from hyperopt import fmin, tpe, hp

# Supposons que data1 et data2 sont vos deux jeux de données
data1 = [1, 2, 3, 4, 5]
data2 = [2, 3, 4, 5, 6]

# Définir la fonction objectif pour comparer les deux jeux de données
def objective_function(params):
    # Ici, vous pouvez utiliser les hyperparamètres pour ajuster votre mesure de similarité ou de différence
    weight1 = params['weight1']
    weight2 = params['weight2']
    
    # Exemple de mesure de similarité ou de différence (ce pourrait être plus complexe selon votre besoin)
    similarity_measure = sum((weight1 * val1 - weight2 * val2) ** 2 for val1, val2 in zip(data1, data2))

    return similarity_measure

# Définir l'espace des hyperparamètres à explorer
space = {
    'weight1': hp.uniform('weight1', 0, 1),
    'weight2': hp.uniform('weight2', 0, 1)
}

# Utiliser l'algorithme TPE pour optimiser la fonction objectif
best = fmin(fn=objective_function, space=space, algo=tpe.suggest, max_evals=100)

print("Meilleurs hyperparamètres :", best)
