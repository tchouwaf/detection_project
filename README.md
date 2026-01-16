# Projet – Prétraitement et analyse des données

## Structure du projet

Sur la branche `master`, le dossier `codes` contient les scripts suivants :

### 📘 code-data-augmentation-simple.ipynb
Notebook dédié au **prétraitement des données** :
- reconnaissance des objets
- création de variables utiles pour la **data augmentation**

### 📘 double_yolo_data_separation.ipynb
Notebook dédié à **la séparation du dataset pour YOLO A et YOLO B** :
- reconnaissance des objets et association aux labels
- création de variables utiles pour la séparation

### 📘 yolo_train.ipynb
Notebook dédié à **L'entrainement des modèles sur les Datasets augmenté et non augmentés** :
- Reconnaissance du GPU
- Entrainement des modèles

### 🐍 labels_preprocessing.py
Script Python pour :
- l’**analyse et l’affichage de la distribution des classes** dans les datasets

### 🐍 train_yolo_a.py
Script Python pour :
- l’entrainement de YOLO A

### 🐍 train_yolo_b.py
Script Python pour :
- l’entrainement de YOLO B


### 🐍 inference.py
Script Python pour :
- l’inférence de YOLO A et YOLO B

## Objectif
Analyser et préparer les données (data augmentation) avant l’entraînement de modèles de détection d’objets.
