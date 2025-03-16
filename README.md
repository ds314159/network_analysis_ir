# Network Analysis and Information Retrieval Package

## Description

Ce package Python offre une solution complète pour l'analyse de réseaux combinant données textuelles et relations structurelles. Conçu pour l'analyse d'articles scientifiques, il intègre plusieurs modules fonctionnels permettant d'explorer un corpus selon différentes dimensions : textuelle, structurelle, thématique et prédictive.

Cette solution a été développée dans le cadre du cours "Analysis of Information Networks" avec pour objectif de créer une architecture logicielle structurée, réutilisable et extensible respectant les bonnes pratiques du génie logiciel.

## Fonctionnalités principales

Le package offre six modules fonctionnels principaux :

1. **Acquisition de données** : Chargement, nettoyage et exploration initiale du corpus
2. **Analyse de graphe** : Construction et analyse de réseaux (citation, co-autorat)
3. **Moteur de recherche** : Indexation et recherche d'information dans le corpus
4. **Clustering** : Regroupement non supervisé de documents par similarité textuelle et structurelle
5. **Classification** : Prédiction supervisée des catégories thématiques
6. **Topic Modeling** : Découverte de thèmes latents avec plusieurs approches (LDA, NMF, BERTopic)

## Structure du projet

```
network_analysis_ir/
│
├── data/                      # Répertoire pour les données brutes et prétraitées
│   ├── raw/                   # Données brutes (JSON, CSV)
│   └── processed/             # Données prétraitées
│
├── src/                       # Code source du projet
│   ├── __init__.py
│   │
│   ├── config/                # Configuration du projet
│   │   ├── __init__.py
│   │   └── config.py          # Paramètres de configuration
│   │
│   ├── data_acquisition/      # Module d'acquisition de données (Fonction 1)
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Chargement des données
│   │   ├── data_cleaner.py    # Nettoyage des données
│   │   └── data_explorer.py   # Exploration des données
│   │
│   ├── graph_analysis/        # Module d'analyse de graphe (Fonction 2)
│   │   ├── __init__.py
│   │   ├── graph_builder.py   # Construction du graphe
│   │   ├── graph_stats.py     # Statistiques sur le graphe
│   │   └── graph_viz.py       # Visualisation du graphe
│   │
│   ├── search_engine/         # Module du moteur de recherche (Fonction 3)
│   │   ├── __init__.py
│   │   ├── indexer.py         # Indexation des documents
│   │   ├── query_processor.py # Traitement des requêtes
│   │   └── ranking.py         # Classement des résultats
│   │
│   ├── clustering/            # Module de clustering (Fonction 4)
│   │   ├── __init__.py
│   │   ├── text_clusterer.py  # Clustering sur le texte
│   │   ├── graph_clusterer.py # Clustering sur le graphe
│   │   └── label_generator.py # Génération d'étiquettes pour les clusters
│   │
│   ├── classification/        # Module de classification (Fonction 5)
│   │   ├── __init__.py
│   │   ├── feature_extractor.py # Extraction de caractéristiques
│   │   ├── classifier.py      # Implémentation des classificateurs
│   │   └── evaluator.py       # Évaluation des performances
│   │
│   ├── topic_modeling/        # Module de Topic Modeling (Fonction 6)
│   │   ├── __init__.py
│   │   ├── lda_model.py       # Modèle LDA
│   │   ├── nmf_model.py       # Modèle NMF
│   │   ├── bert_model.py      # Modèle BERTopic
│   │   ├── topic_extractor.py # Gestion des modèles de Topic Modeling
│   │   ├── topic_visualisation.py # Visualisation des topics
│   │   └── topic_evaluator.py # Évaluation des modèles
│   │
│   └── utils/                 # Utilitaires partagés
│       ├── __init__.py
│       ├── text_utils.py      # Fonctions utilitaires pour le texte
│       ├── viz_utils.py       # Fonctions utilitaires pour la visualisation
│       └── io_utils.py        # Fonctions utilitaires pour l'entrée/sortie
│
├── notebooks/                 # Notebooks Jupyter pour démonstration et expérimentation
│   ├── 1_data_exploration.ipynb
│   ├── 2_graph_analysis.ipynb
│   ├── 3_search_engine.ipynb
│   ├── 4_clustering.ipynb
│   ├── 5_classification.ipynb
│   └── 6_topic_modeling.ipynb
│
├── tests/                     # Tests unitaires et d'intégration
│   ├── __init__.py
│   ├── test_data_acquisition.py
│   ├── test_graph_analysis.py
│   ├── test_search_engine.py
│   ├── test_clustering.py
│   ├── test_classification.py
│   └── test_topic_modeling.py
│
├── app/                       # Extension pour une application web (future implémentation)
│   ├── __init__.py
│   ├── app.py                 # Point d'entrée de l'application web
│   ├── static/                # Fichiers statiques (CSS, JS)
│   └── templates/             # Templates HTML
│
├── requirements.txt           # Dépendances du projet
├── setup.py                   # Configuration du package
└── README.md                  # Documentation du projet
```

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/network_analysis_ir.git
cd network_analysis_ir

# Installer les dépendances
pip install -r requirements.txt

# Installer le package en mode développement
pip install -e .
```

## Utilisation

Le package peut être utilisé de plusieurs façons :

### Via les notebooks Jupyter

Des notebooks Jupyter sont fournis dans le répertoire `notebooks/` pour démontrer l'utilisation de chaque module fonctionnel :

```bash
jupyter notebook notebooks/
```

### Import des modules dans votre code

```python
# Exemple d'utilisation du module d'acquisition de données
from network_analysis_ir.src.data_acquisition.data_loader import DataLoader
from network_analysis_ir.src.data_acquisition.data_cleaner import DataCleaner

# Chargement et nettoyage des données
loader = DataLoader()
data = loader.load_json("path/to/data.json")
cleaner = DataCleaner()
cleaned_data = cleaner.clean_text(data)

# Exemple d'utilisation du module d'analyse de graphe
from network_analysis_ir.src.graph_analysis.graph_builder import GraphBuilder

# Construction du graphe de citation
graph_builder = GraphBuilder(cleaned_data)
citation_graph = graph_builder.build_citation_graph()
```

## Modules détaillés

### 1. Module d'acquisition de données

Ce module gère le chargement, le nettoyage et l'exploration initiale des données :

- **DataLoader** : Importation des données depuis divers formats (JSON, CSV)
- **DataCleaner** : Fonctions de prétraitement textuel (suppression des stop words, lemmatisation)
- **DataExplorer** : Analyse statistique descriptive et visualisations du corpus

### 2. Module d'analyse de graphe

Ce module permet la modélisation et l'analyse du corpus sous forme de graphe :

- **GraphBuilder** : Construction de différents types de graphes (citation, co-autorat, biparti)
- **GraphStats** : Calcul de métriques topologiques (centralité, clustering, etc.)
- **GraphVisualizer** : Visualisation des graphes et des communautés

### 3. Module de moteur de recherche

Ce module implémente un système de recherche d'information dans le corpus :

- **Indexer** : Construction d'index inversés et vectoriels (TF-IDF)
- **QueryProcessor** : Traitement et normalisation des requêtes
- **RankingSystem** : Classement des résultats par pertinence

### 4. Module de clustering

Ce module permet d'identifier des regroupements naturels dans le corpus :

- **TextClusterer** : Clustering basé sur la similarité textuelle (K-means, DBSCAN)
- **GraphClusterer** : Détection de communautés dans le graphe (Louvain, spectral)
- **LabelGenerator** : Extraction d'étiquettes représentatives pour les clusters

### 5. Module de classification

Ce module permet de prédire la catégorie thématique des documents :

- **FeatureExtractor** : Extraction de caractéristiques textuelles et structurelles
- **Classifier** : Implémentation de divers classificateurs (SVM, Random Forest)
- **Evaluator** : Mesures de performance et validation des modèles

### 6. Module de Topic Modeling

Ce module permet de découvrir les thèmes latents présents dans le corpus :

- **LDAModel** : Modélisation par Allocation de Dirichlet Latente
- **NMFModel** : Factorisation de Matrices Non-négatives
- **BERTopicModel** : Approche basée sur les embeddings contextuels
- **TopicExtractor** : Interface unifiée pour les différents modèles
- **TopicVisualization** : Visualisation des thèmes découverts
- **TopicEvaluator** : Évaluation de la qualité des thèmes

## Dépendances principales

- numpy, pandas, scipy : Manipulation et analyse de données
- nltk, spacy : Traitement du langage naturel
- networkx, community : Analyse de graphes et détection de communautés
- scikit-learn : Apprentissage automatique
- gensim : Topic modeling
- matplotlib, seaborn, plotly : Visualisation de données
- sentence-transformers, bertopic : Embeddings contextuels et topic modeling avancé

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Auteur

Mehdi Mansour - [mehdi.mansour@univ-lyon2.fr](mailto:mehdi.mansour@univ-lyon2.fr)

## Remerciements

Ce projet a été développé dans le cadre du cours "Analysis of Information Networks" à l'ICOM, Université Lyon 2.
