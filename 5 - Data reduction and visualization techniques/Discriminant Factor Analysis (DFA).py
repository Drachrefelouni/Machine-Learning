#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:07:26 2024

@author: achref
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Charger les données d'exemple (iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Appliquer l'AFD
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Afficher les données projetées dans l'espace AFD
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], label=f'Classe {label}', alpha=0.8)
plt.title('Analyse Factorielle Discriminante (AFD)')
plt.xlabel('Premier axe discriminant')
plt.ylabel('Deuxième axe discriminant')
plt.legend()
plt.grid(True)
plt.show()
