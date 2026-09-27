# Linear Regression — Cours et Homework (Machine Learning Zoomcamp)

Ce dossier regroupe deux notebooks distincts du module régression linéaire du **Machine Learning Zoomcamp**, plus les deux datasets utilisés.

## Contenu

| Fichier | Rôle |
|---|---|
| `data_prep.ipynb` | **Notebook du cours** — reproduit pas à pas les commandes vues en leçon : nettoyage des données, régression linéaire par équation normale, encodage one-hot des variables catégorielles, régularisation. Basé sur `data.csv` (dataset de prix de voitures) |
| `homework2.ipynb` | **Devoir (homework 2)** — répond aux questions du module en appliquant les mêmes techniques sur un nouveau dataset, `car_fuel_efficiency_2026.csv` |
| `data.csv` | Dataset de prix de voitures (Car Features and MSRP), utilisé dans `data_prep.ipynb` |
| `car_fuel_efficiency_2026.csv` | Dataset de consommation de carburant des voitures, utilisé dans `homework2.ipynb` |

## `data_prep.ipynb` — le cours

Reprend dans l'ordre les notions du module :

1. Nettoyage des colonnes et des valeurs textuelles
2. Régression linéaire vue comme un produit scalaire, puis une multiplication matrice-vecteur
3. Dérivation et implémentation de l'équation normale : `w = (XᵀX)⁻¹Xᵀy`
4. Ajout de features numériques dérivées (ex. `age`)
5. Encodage one-hot des variables catégorielles (`make`, `transmission_type`, etc.)
6. Découverte du problème de colinéarité en encodant toutes les catégories d'un coup (RMSE qui explose)
7. Correction par régularisation (`XᵀX + r·I`)

## `homework2.ipynb` — le devoir

Utilise `car_fuel_efficiency_2026.csv`, réduit aux colonnes `engine_displacement`, `horsepower`, `vehicle_weight`, `model_year`, `fuel_efficiency_mpg` (cible à prédire).

Répond aux questions suivantes :

1. **EDA** : distribution de `fuel_efficiency_mpg`, présence d'une longue traîne
2. **Valeurs manquantes** : comparaison de deux stratégies (remplir avec 0 vs avec la moyenne du train), en calculant la moyenne uniquement sur `df_train` pour éviter toute fuite de données
3. **Régularisation** : test de plusieurs valeurs de `r` (`0, 0.01, 0.1, 1, 5, 10, 100`), évaluation sur validation
4. **Stabilité du split** : test de 10 seeds différents, calcul de l'écart-type des RMSE obtenus
5. **Modèle final** : entraînement sur train + validation combinés (seed=9), évaluation sur test avec `r=0.001`

## Comment lancer

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook data_prep.ipynb      # pour le cours
jupyter notebook homework2.ipynb      # pour le devoir
```

## Fait dans le cadre de

[Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) — DataTalksClub.
