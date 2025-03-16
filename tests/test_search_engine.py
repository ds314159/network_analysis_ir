# tests/test_search_engine.py

import unittest
import os
import sys
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

# Ajout du répertoire parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import des classes à tester
from src.search_engine.indexer import Indexer
from src.search_engine.query_processor import QueryProcessor
from src.search_engine.ranking import RankingSystem
from src.config.config import load_config


class TestIndexer(unittest.TestCase):
    def setUp(self):
        # Créer une configuration de test
        self.config = {"preprocessing": {"min_df": 1, "max_df": 0.95}}

        # Créer des documents de test
        self.test_docs = [
            {
                "id": "1",
                "title": "Document sur le traitement du langage naturel",
                "abstract": "Cet article traite du traitement automatique du langage naturel et de ses applications.",
                "authors": ["Auteur A", "Auteur B"],
                "year": 2020,
                "venue": "Conférence NLP",
                "class": 1
            },
            {
                "id": "2",
                "title": "Algorithmes d'apprentissage automatique",
                "abstract": "Cet article présente plusieurs algorithmes d'apprentissage automatique.",
                "authors": ["Auteur C"],
                "year": 2019,
                "venue": "Journal ML",
                "class": 2
            },
            {
                "id": "3",
                "title": "Réseaux de neurones et NLP",
                "abstract": "Les réseaux de neurones sont utilisés pour le traitement du langage naturel.",
                "authors": ["Auteur D"],
                "year": 2021,
                "venue": "Conférence IA",
                "class": 1
            }
        ]

        # Créer un indexeur
        self.indexer = Indexer(self.config)

    def test_build_index(self):
        # Construire l'index
        stats = self.indexer.build_index(self.test_docs, content_field='abstract', id_field='id', min_df=1, max_df=1.0)

        # Vérifier que l'index est correctement construit
        self.assertEqual(stats['num_documents'], 3)
        self.assertTrue(stats['vocabulary_size'] > 0)
        self.assertIsNotNone(self.indexer.tfidf_matrix)
        self.assertIsNotNone(self.indexer.doc_term_matrix)
        self.assertIsNotNone(self.indexer.feature_names)
        self.assertEqual(len(self.indexer.document_ids), 3)

    def test_get_term_frequencies(self):
        # Construire l'index d'abord
        self.indexer.build_index(self.test_docs, content_field='abstract', min_df=1, max_df=1.0)

        # Obtenir les fréquences des termes
        term_freqs = self.indexer.get_term_frequencies(top_n=5)

        # Vérifier que le résultat est un dictionnaire non vide
        self.assertIsInstance(term_freqs, dict)
        self.assertTrue(len(term_freqs) > 0)
        self.assertTrue(len(term_freqs) <= 5)  # Au maximum 5 termes

    def test_get_document_vector(self):
        # Construire l'index d'abord
        self.indexer.build_index(self.test_docs, content_field='abstract', min_df=1, max_df=1.0)

        # Obtenir le vecteur d'un document
        vector_tfidf = self.indexer.get_document_vector("1", use_tfidf=True)
        vector_tf = self.indexer.get_document_vector("1", use_tfidf=False)

        # Vérifier que les vecteurs ont la bonne forme
        self.assertIsInstance(vector_tfidf, np.ndarray)
        self.assertIsInstance(vector_tf, np.ndarray)
        self.assertEqual(len(vector_tfidf), len(self.indexer.feature_names))
        self.assertEqual(len(vector_tf), len(self.indexer.feature_names))

        # Vérifier qu'une erreur est levée pour un ID inexistant
        with self.assertRaises(ValueError):
            self.indexer.get_document_vector("999")


class TestQueryProcessor(unittest.TestCase):
    def setUp(self):
        # Créer une configuration de test
        self.config = {"preprocessing": {"min_df": 1, "max_df": 0.95}}

        # Créer des documents de test
        self.test_docs = [
            {
                "id": "1",
                "title": "Document sur le traitement du langage naturel",
                "abstract": "Cet article traite du traitement automatique du langage naturel et de ses applications.",
                "class": 1
            },
            {
                "id": "2",
                "title": "Algorithmes d'apprentissage automatique",
                "abstract": "Cet article présente plusieurs algorithmes d'apprentissage automatique.",
                "class": 2
            }
        ]

        # Créer un indexeur et construire l'index
        self.indexer = Indexer(self.config)
        self.indexer.build_index(self.test_docs, content_field='abstract', min_df=1, max_df=1.0)

        # Créer un processeur de requêtes
        self.query_processor = QueryProcessor(self.indexer)

    def test_process_query(self):
        # Traiter une requête
        query_vector_tfidf = self.query_processor.process_query("traitement du langage naturel", use_tfidf=True)
        query_vector_tf = self.query_processor.process_query("traitement du langage naturel", use_tfidf=False)

        # Vérifier que les vecteurs de requête ont la bonne forme
        self.assertIsInstance(query_vector_tfidf, np.ndarray)
        self.assertIsInstance(query_vector_tf, np.ndarray)
        self.assertEqual(len(query_vector_tfidf), len(self.indexer.feature_names))
        self.assertEqual(len(query_vector_tf), len(self.indexer.feature_names))

    # Note: Le test de semantic_embedding nécessite la bibliothèque sentence-transformers,
    # qui peut être lourde pour les tests unitaires. Vous pourriez vouloir le désactiver
    # ou le mocker dans un environnement de test réel.
    def test_semantic_embedding_basic(self):
        # Tester uniquement la signature de la méthode sans exécuter le code réel
        # Cette approche est utile pour les tests unitaires légers
        try:
            # Vérifier que la méthode est correctement définie
            self.assertTrue(callable(getattr(self.query_processor, 'semantic_embedding')))
        except AttributeError:
            self.fail("La méthode semantic_embedding n'existe pas")


class TestRankingSystem(unittest.TestCase):
    def setUp(self):
        # Créer une configuration de test
        self.config = {"preprocessing": {"min_df": 1, "max_df": 0.95}}

        # Créer des documents de test
        self.test_docs = [
            {
                "id": "1",
                "title": "Document sur le traitement du langage naturel",
                "abstract": "Cet article traite du traitement automatique du langage naturel et de ses applications.",
                "authors": ["Auteur A", "Auteur B"],
                "year": 2020,
                "venue": "Conférence NLP",
                "class": 1
            },
            {
                "id": "2",
                "title": "Algorithmes d'apprentissage automatique",
                "abstract": "Cet article présente plusieurs algorithmes d'apprentissage automatique.",
                "authors": ["Auteur C"],
                "year": 2019,
                "venue": "Journal ML",
                "class": 2
            },
            {
                "id": "3",
                "title": "Réseaux de neurones et NLP",
                "abstract": "Les réseaux de neurones sont utilisés pour le traitement du langage naturel.",
                "authors": ["Auteur D"],
                "year": 2021,
                "venue": "Conférence IA",
                "class": 1
            }
        ]

        # Créer un indexeur et construire l'index
        self.indexer = Indexer(self.config)
        self.indexer.build_index(self.test_docs, content_field='abstract', min_df=1, max_df=1.0)

        # Créer un système de classement
        self.ranking_system = RankingSystem(self.indexer, self.test_docs)

        # Créer un processeur de requêtes
        self.query_processor = QueryProcessor(self.indexer)





if __name__ == '__main__':
    unittest.main()