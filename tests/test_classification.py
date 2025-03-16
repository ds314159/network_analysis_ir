import unittest
import os
import numpy as np
import tempfile
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Ajouter le répertoire parent au chemin de recherche
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer les classes à tester
from src.classification.classifier import Classifier
from src.classification.evaluator import Evaluator
from src.classification.feature_extractor import FeatureExtractor



# Simuler un indexeur avec une matrice TF-IDF factice
class MockIndexer:
    def __init__(self):
        self.document_ids = ["doc_1", "doc_2", "doc_3", "doc_4"]
        self.tfidf_matrix = np.array([[0.1, 0.3, 0.6],
                                      [0.5, 0.2, 0.3],
                                      [0.4, 0.5, 0.1],
                                      [0.2, 0.1, 0.7]])
        self.doc_term_matrix = self.tfidf_matrix
        self.feature_names = ["word1", "word2", "word3"]


# Simuler un graphe de citations (petit graphe factice)
import networkx as nx


class MockGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([("doc_1", "doc_2"), ("doc_2", "doc_3"), ("doc_3", "doc_4")])

    def __contains__(self, node):
        return node in self.graph.nodes


# Instancier les mocks
mock_indexer = MockIndexer()
mock_graph = MockGraph()

# Instancier le FeatureExtractor avec des données factices
feature_extractor = FeatureExtractor(indexer=mock_indexer, graph=mock_graph.graph)


class TestClassifier(unittest.TestCase):
    def setUp(self):
        """Créer un classificateur pour les tests."""
        self.classifier = Classifier(feature_extractor)
        self.doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4"]
        self.labels = ["A", "B", "A", "B"]

    def test_train_classifier_svm(self):
        """Tester l'entraînement d'un SVM."""
        model = self.classifier.train_classifier(self.doc_ids, self.labels, classifier_type='svm')
        self.assertIsInstance(model, SVC)
        self.assertTrue(hasattr(model, "predict"))

    def test_train_classifier_random_forest(self):
        """Tester l'entraînement d'un Random Forest."""
        model = self.classifier.train_classifier(self.doc_ids, self.labels, classifier_type='rf')
        self.assertIsInstance(model, RandomForestClassifier)

    def test_train_classifier_logistic_regression(self):
        """Tester l'entraînement d'une régression logistique."""
        model = self.classifier.train_classifier(self.doc_ids, self.labels, classifier_type='lr')
        self.assertIsInstance(model, LogisticRegression)

    def test_predict(self):
        """Tester la prédiction avec un modèle entraîné."""
        self.classifier.train_classifier(self.doc_ids, self.labels, classifier_type='svm')
        predictions = self.classifier.predict(self.doc_ids)
        self.assertEqual(len(predictions), len(self.doc_ids))

    def test_predict_proba(self):
        """Tester la prédiction des probabilités."""
        self.classifier.train_classifier(self.doc_ids, self.labels, classifier_type='svm')
        probas = self.classifier.predict_proba(self.doc_ids)
        self.assertEqual(len(probas), len(self.doc_ids))


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        """Créer un évaluateur pour tester la classification."""
        self.evaluator = Evaluator()
        self.true_labels = ["A", "B", "A", "B"]
        self.predicted_labels = ["A", "B", "A", "A"]

    def test_evaluate_classifier(self):
        """Tester l'évaluation du classificateur."""
        results = self.evaluator.evaluate_classifier(self.true_labels, self.predicted_labels)
        self.assertIn("accuracy", results)
        self.assertIn("classification_report", results)
        self.assertIn("confusion_matrix", results)

    def test_cross_validate(self):
        """Tester la validation croisée."""
        classifier = Classifier(feature_extractor)
        classifier.train_classifier(["doc_1", "doc_2", "doc_3", "doc_4"], ["A", "B", "A", "B"], classifier_type='svm')

        results = self.evaluator.cross_validate(classifier, self.true_labels, self.predicted_labels, n_splits=3)
        self.assertIn("cv_accuracy_mean", results)


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        """Initialiser l'extracteur de caractéristiques."""
        self.feature_extractor = FeatureExtractor(indexer=mock_indexer, graph=mock_graph.graph)

    def test_extract_text_features(self):
        """Tester l'extraction de caractéristiques textuelles."""
        features = self.feature_extractor.extract_text_features(["doc_1", "doc_2"])
        self.assertEqual(features.shape, (2, 3))  # 2 documents, 3 features (mots)

    def test_extract_graph_features(self):
        """Tester l'extraction de caractéristiques de graphe."""
        features = self.feature_extractor.extract_graph_features(["doc_1", "doc_2"])
        self.assertEqual(features.shape[0], 2)  # 2 documents

    def test_extract_node2vec_features(self):
        """Tester l'extraction de Node2Vec embeddings."""
        embeddings = self.feature_extractor.extract_node2vec_features(["doc_1", "doc_2"], dimensions=10)
        self.assertEqual(embeddings.shape, (2, 10))  # 2 documents, 10 dimensions

    def test_combine_features(self):
        """Tester la combinaison des caractéristiques texte + graphe."""
        text_features = np.array([[0.1, 0.2], [0.3, 0.4]])
        graph_features = np.array([[0.5, 0.6], [0.7, 0.8]])
        combined = self.feature_extractor.combine_features(text_features, graph_features)
        self.assertEqual(combined.shape[1], 4)  # 2+2 = 4 features combinés


if __name__ == '__main__':
    unittest.main()
