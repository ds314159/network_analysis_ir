# tests/test_data_acquisition.py

import unittest
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import sys

# Ajout du répertoire parent au chemin de recherche
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import des classes à tester
from src.config.config import load_config
from src.data_acquisition.data_loader import DataLoader
from src.data_acquisition.data_cleaner import DataCleaner
from src.data_acquisition.data_explorer import DataExplorer

import nltk
nltk.download('stopwords')

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Créer une configuration de test
        self.config = {"preprocessing": {"min_word_length": 3}}

        # Créer des données de test
        self.test_data = [
            {
                "id": "1",
                "title": "Test Document 1",
                "abstract": "This is a test abstract for document 1.",
                "authors": ["Author A", "Author B"],
                "year": 2020,
                "venue": "Test Conference",
                "references": ["2", "3"],
                "class": 1
            },
            {
                "id": "2",
                "title": "Test Document 2",
                "abstract": "This is a test abstract for document 2.",
                "authors": ["Author C"],
                "year": 2021,
                "venue": "Test Journal",
                "references": [],
                "class": 2
            }
        ]

        # Créer un fichier CSV temporaire pour les tests
        self.temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        pd.DataFrame({
            'id': ['1', '2'],
            'title': ['Test Document 1', 'Test Document 2'],
            'abstract': ['This is a test abstract for document 1.', 'This is a test abstract for document 2.'],
            'authors': [["Author A", "Author B"], ["Author C"]],
            'year': [2020, 2021],
            'venue': ['Test Conference', 'Test Journal'],
            'references': [["2", "3"], []],
            'class': [1, 2]
        }).to_csv(self.temp_csv.name, index=False, sep='\t')

        # Créer un fichier JSON temporaire pour les tests
        self.temp_json = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        with open(self.temp_json.name, 'w') as f:
            json.dump(self.test_data, f)



    def test_load_from_csv(self):
        loader = DataLoader(self.config)
        data = loader.load_from_csv(self.temp_csv.name)

        # Vérifier que les données sont correctement chargées
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 2)
        self.assertEqual(data.iloc[0]['title'], 'Test Document 1')

    def test_load_from_json(self):
        loader = DataLoader(self.config)
        data = loader.load_from_json(self.temp_json.name)

        # Vérifier que les données sont correctement chargées
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['title'], 'Test Document 1')


class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        # Créer une configuration de test
        self.config = {"preprocessing": {"min_word_length": 3}}
        self.cleaner = DataCleaner(self.config)

    def test_clean_text(self):
        # Texte à nettoyer
        text = "Hello, World! This is a TEST."

        # Nettoyer le texte
        cleaned_text = self.cleaner.clean_text(text)

        # Vérifier le résultat
        self.assertEqual(cleaned_text, "hello world this is a test")

    def test_remove_stop_words(self):
        # Liste de tokens avec des mots vides
        tokens = ["hello", "the", "world", "is", "beautiful"]

        # Supprimer les mots vides
        filtered_tokens = self.cleaner.remove_stop_words(tokens)

        # Vérifier que les mots vides ont été supprimés
        self.assertNotIn("the", filtered_tokens)
        self.assertNotIn("is", filtered_tokens)
        self.assertIn("hello", filtered_tokens)
        self.assertIn("world", filtered_tokens)
        self.assertIn("beautiful", filtered_tokens)

    def test_normalize_text(self):
        # Texte à normaliser
        text = "running runs runner"

        # Normaliser avec stemming
        stemmed = self.cleaner.normalize_text(text, method='stemming')

        # Normaliser avec lemmatisation
        lemmatized = self.cleaner.normalize_text(text, method='lemmatization')

        # Vérifier les résultats
        self.assertIn("run", stemmed)
        self.assertIn("running", lemmatized)  # lemmatizer conserve "running" tel quel


class TestDataExplorer(unittest.TestCase):
    def setUp(self):
        # Créer des données de test
        self.test_data = [
            {
                "id": "1",
                "title": "Test Document 1",
                "abstract": "This is a test abstract for document 1.",
                "authors": ["Author A", "Author B"],
                "year": 2020,
                "venue": "Test Conference",
                "references": ["2", "3"],
                "class": 1
            },
            {
                "id": "2",
                "title": "Test Document 2",
                "abstract": "This is a test abstract for document 2.",
                "authors": ["Author C"],
                "year": 2021,
                "venue": "Test Journal",
                "references": [],
                "class": 2
            }
        ]

        self.explorer = DataExplorer(self.test_data)

    def test_get_basic_stats(self):
        # Obtenir les statistiques de base
        stats = self.explorer.get_basic_stats()

        # Vérifier les résultats
        self.assertEqual(stats['num_documents'], 2)
        self.assertEqual(stats['num_authors'], 3)
        self.assertTrue('avg_text_length' in stats)
        self.assertTrue('class_distribution' in stats)
        self.assertTrue('temporal_distribution' in stats)

    def test_get_class_distribution(self):
        # Obtenir la distribution des classes
        class_dist = self.explorer._get_class_distribution()

        # Vérifier les résultats
        self.assertEqual(class_dist[1], 1)
        self.assertEqual(class_dist[2], 1)

    def test_get_temporal_distribution(self):
        # Obtenir la distribution temporelle
        temp_dist = self.explorer._get_temporal_distribution()

        # Vérifier les résultats
        self.assertEqual(temp_dist[2020], 1)
        self.assertEqual(temp_dist[2021], 1)


if __name__ == '__main__':
    unittest.main()