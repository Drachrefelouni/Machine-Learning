#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:51:53 2024

@author: achref
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Charger le jeu de données Iris
iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

# Ajouter des valeurs manquantes artificielles dans quelques cellules
iris_df.iloc[2, 1] = np.nan
iris_df.iloc[4, 3] = np.nan
iris_df.iloc[7, 2] = np.nan

# Imputation par la moyenne
imputer = SimpleImputer(strategy='mean')
iris_df_imputed = pd.DataFrame(imputer.fit_transform(iris_df), columns=iris_df.columns)

# Diviser les caractéristiques et les étiquettes
X = iris_df_imputed.drop(columns=['target'])  # Caractéristiques
Y = iris_df_imputed['target']  # Étiquettes

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalisation (Min-Max Scaling)
scaler = MinMaxScaler()

# Adapter et transformer les données d'entraînement
X_train_normalized = scaler.fit_transform(X_train)

# Transformer les données de test (utiliser seulement transform, pas fit_transform)
X_test_normalized = scaler.transform(X_test)

# Afficher les caractéristiques normalisées
print("Caractéristiques normalisées :\n", X_train_normalized)
