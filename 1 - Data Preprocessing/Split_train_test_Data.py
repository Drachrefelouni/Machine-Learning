#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:48:43 2024

@author: achref
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

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

# Afficher les tailles des ensembles d'entraînement et de test
print("Taille de l'ensemble d'entraînement :", X_train.shape)
print("Taille de l'ensemble de test :", X_test.shape)
