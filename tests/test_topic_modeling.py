# tests/test_topic_modeling.py

import unittest
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ajout du répertoire parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import des classes à tester
from src.topic_modeling.lda_model import LDATopicModel
from src.topic_modeling.nmf_model import NMFTopicModel
from src.topic_modeling.bert_model import BERTopicModel
from src.topic_modeling.topic_extractor import TopicExtractor
from src.topic_modeling.topic_evaluator import evaluate_topic_coherence


class TestLDATopicModel(unittest.TestCase):
    def setUp(self):
        # Créer des documents de test
        self.test_docs = [
            "machine learning algorithms can be used for classification tasks",
            "natural language processing helps computers understand human language",
            "deep learning models require large amounts of data",
            "support vector machines are effective for classification",
            "neural networks can learn complex patterns in data"
        ]

        # Initialiser le modèle LDA
        self.lda_model = LDATopicModel(num_topics=2, max_iter=5)

    def test_initialization(self):
        # Vérifier que le modèle est correctement initialisé
        self.assertEqual(self.lda_model.num_topics, 2)
        self.assertEqual(self.lda_model.max_iter, 5)
        self.assertIsNotNone(self.lda_model.model)
        self.assertIsNotNone(self.lda_model.vectorizer)

    def test_fit_transform(self):
        # Entraîner le modèle
        self.lda_model.fit(self.test_docs)

        # Vérifier que la matrice document-terme est créée
        self.assertIsNotNone(getattr(self.lda_model, 'doc_term_matrix', None))

        # Vérifier que la transformation fonctionne
        topic_distribution = self.lda_model.transform(self.test_docs)
        self.assertEqual(topic_distribution.shape[0], len(self.test_docs))
        self.assertEqual(topic_distribution.shape[1], self.lda_model.num_topics)

        # Vérifier que les probabilités somment à 1 pour chaque document
        for doc_probs in topic_distribution:
            self.assertAlmostEqual(sum(doc_probs), 1.0, places=5)

    def test_get_topics(self):
        # Entraîner le modèle
        self.lda_model.fit(self.test_docs)

        # Récupérer les topics
        topics = self.lda_model.get_topics(num_words=5)

        # Vérifier la structure des topics
        self.assertEqual(len(topics), self.lda_model.num_topics)
        for topic_id, words in topics.items():
            self.assertTrue(topic_id.startswith("Topic "))
            self.assertEqual(len(words), 5)

            # Vérifier que les mots sont des chaînes de caractères
            for word in words:
                self.assertIsInstance(word, str)


class TestNMFTopicModel(unittest.TestCase):
    def setUp(self):
        # Créer des documents de test
        self.test_docs = [
            "machine learning algorithms can be used for classification tasks",
            "natural language processing helps computers understand human language",
            "deep learning models require large amounts of data",
            "support vector machines are effective for classification",
            "neural networks can learn complex patterns in data"
        ]

        # Initialiser le modèle NMF
        self.nmf_model = NMFTopicModel(num_topics=2)

    def test_initialization(self):
        # Vérifier que le modèle est correctement initialisé
        self.assertEqual(self.nmf_model.num_topics, 2)
        self.assertIsNotNone(self.nmf_model.model)
        self.assertIsNotNone(self.nmf_model.vectorizer)

    def test_fit_transform(self):
        # Entraîner le modèle
        self.nmf_model.fit(self.test_docs)

        # Vérifier que la matrice document-terme est créée
        self.assertIsNotNone(getattr(self.nmf_model, 'doc_term_matrix', None))

        # Vérifier que la transformation fonctionne
        topic_distribution = self.nmf_model.transform(self.test_docs)
        self.assertEqual(topic_distribution.shape[0], len(self.test_docs))
        self.assertEqual(topic_distribution.shape[1], self.nmf_model.num_topics)

        # Pour NMF les valeurs doivent être non-négatives
        self.assertTrue(np.all(topic_distribution >= 0))

    def test_get_topics(self):
        # Entraîner le modèle
        self.nmf_model.fit(self.test_docs)

        # Récupérer les topics
        topics = self.nmf_model.get_topics(num_words=5)

        # Vérifier la structure des topics
        self.assertEqual(len(topics), self.nmf_model.num_topics)
        for topic_id, words in topics.items():
            self.assertTrue(topic_id.startswith("Topic "))
            self.assertEqual(len(words), 5)

            # Vérifier que les mots sont des chaînes de caractères
            for word in words:
                self.assertIsInstance(word, str)





class TestTopicExtractor(unittest.TestCase):
    def setUp(self):
        # Créer des documents de test
        self.test_docs = [
            "machine learning algorithms can be used for classification tasks",
            "natural language processing helps computers understand human language",
            "deep learning models require large amounts of data",
            "support vector machines are effective for classification",
            "neural networks can learn complex patterns in data"
        ]

    def test_lda_integration(self):
        # Tester l'intégration avec LDA
        extractor = TopicExtractor(model_type='lda', num_topics=2)
        extractor.fit(self.test_docs)

        # Récupérer les topics
        topics = extractor.get_topics(num_words=5)

        # Vérifier la structure
        self.assertEqual(len(topics), 2)
        for topic_id, words in topics.items():
            self.assertTrue(topic_id.startswith("Topic "))
            self.assertEqual(len(words), 5)

        # Tester transform
        topic_distribution = extractor.transform(self.test_docs)
        self.assertEqual(topic_distribution.shape[0], len(self.test_docs))
        self.assertEqual(topic_distribution.shape[1], 2)

    def test_nmf_integration(self):
        # Tester l'intégration avec NMF
        extractor = TopicExtractor(model_type='nmf', num_topics=2)
        extractor.fit(self.test_docs)

        # Récupérer les topics
        topics = extractor.get_topics(num_words=5)

        # Vérifier la structure
        self.assertEqual(len(topics), 2)
        for topic_id, words in topics.items():
            self.assertTrue(topic_id.startswith("Topic "))
            self.assertEqual(len(words), 5)

        # Tester transform
        topic_distribution = extractor.transform(self.test_docs)
        self.assertEqual(topic_distribution.shape[0], len(self.test_docs))
        self.assertEqual(topic_distribution.shape[1], 2)

    def test_invalid_model_type(self):
        # Tester avec un type de modèle invalide
        with self.assertRaises(ValueError):
            extractor = TopicExtractor(model_type='invalid_type')


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # Créer des documents de test
        self.test_docs = [
            "machine learning algorithms can be used for classification tasks",
            "natural language processing helps computers understand human language",
            "deep learning models require large amounts of data",
            "support vector machines are effective for classification",
            "neural networks can learn complex patterns in data"
        ]

        # Initialiser les modèles
        self.lda_model = LDATopicModel(num_topics=2, max_iter=5)
        self.nmf_model = NMFTopicModel(num_topics=2)

        # Entraîner les modèles
        self.lda_model.fit(self.test_docs)
        self.nmf_model.fit(self.test_docs)

    def test_evaluate_topic_coherence(self):
        # Évaluer la cohérence du modèle LDA
        lda_coherence = evaluate_topic_coherence(self.lda_model, self.test_docs)

        # Vérifier que le score est dans l'intervalle [-1, 1]
        self.assertGreaterEqual(lda_coherence, -1.0)
        self.assertLessEqual(lda_coherence, 1.0)

        # Évaluer la cohérence du modèle NMF
        nmf_coherence = evaluate_topic_coherence(self.nmf_model, self.test_docs)

        # Vérifier que le score est dans l'intervalle [-1, 1]
        self.assertGreaterEqual(nmf_coherence, -1.0)
        self.assertLessEqual(nmf_coherence, 1.0)


if __name__ == '__main__':
    unittest.main()