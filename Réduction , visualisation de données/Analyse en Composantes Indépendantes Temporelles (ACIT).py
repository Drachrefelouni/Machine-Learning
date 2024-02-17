#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:40:56 2024

@author: achref
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import FastICA

# Générer des données temporelles fictives
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal sinusoïdal
s2 = np.sign(np.sin(3 * time))  # Signal carré
s3 = np.random.randn(n_samples)  # Bruit gaussien

S = np.c_[s1, s2, s3]

# Mélanger les signaux
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Matrice de mélange
X = np.dot(S, A.T)  # Mélange de signaux

# Appliquer l'ACIT avec FastICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)

# Tracer les signaux originaux et les signaux estimés
plt.figure(figsize=(12, 6))

plt.subplot(3, 2, 1)
plt.plot(time, S[:, 0], 'r')
plt.title('Signal original 1')

plt.subplot(3, 2, 2)
plt.plot(time, S_[:, 0], 'b')
plt.title('Signal estimé 1')

plt.subplot(3, 2, 3)
plt.plot(time, S[:, 1], 'r')
plt.title('Signal original 2')

plt.subplot(3, 2, 4)
plt.plot(time, S_[:, 1], 'b')
plt.title('Signal estimé 2')

plt.subplot(3, 2, 5)
plt.plot(time, S[:, 2], 'r')
plt.title('Signal original 3')

plt.subplot(3, 2, 6)
plt.plot(time, S_[:, 2], 'b')
plt.title('Signal estimé 3')

plt.tight_layout()
plt.show()
