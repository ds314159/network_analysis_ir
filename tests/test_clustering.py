import unittest
import os
import json
import networkx as nx
import numpy as np
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer

# Ajouter le répertoire parent au chemin de recherche
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer les classes à tester
from src.clustering.graph_clusterer import GraphClusterer
from src.clustering.text_clusterer import TextClusterer
from src.clustering.label_generator import LabelGenerator



class TestGraphClusterer(unittest.TestCase):
    def setUp(self):
        """Créer un graphe d'exemple (Karate Club) pour les tests."""
        self.graph = nx.karate_club_graph()
        self.clusterer = GraphClusterer(self.graph)

    def test_cluster_louvain(self):
        """Tester l'algorithme de Louvain."""
        clusters = self.clusterer.cluster_louvain()
        self.assertTrue(clusters, "L'algorithme de Louvain n'a détecté aucun cluster.")
        self.assertEqual(sum(len(nodes) for nodes in clusters.values()), self.graph.number_of_nodes())

    def test_cluster_spectral(self):
        """Tester le clustering spectral."""
        clusters = self.clusterer.cluster_spectral(n_clusters=4)
        self.assertTrue(clusters, "L'algorithme Spectral Clustering n'a détecté aucun cluster.")
        self.assertEqual(sum(len(nodes) for nodes in clusters.values()), self.graph.number_of_nodes())

    def test_cluster_label_propagation(self):
        """Tester l'algorithme Label Propagation."""
        clusters = self.clusterer.cluster_label_propagation()
        self.assertTrue(clusters, "L'algorithme Label Propagation n'a détecté aucun cluster.")
        self.assertEqual(sum(len(nodes) for nodes in clusters.values()), self.graph.number_of_nodes())

    def test_cluster_leiden(self):
        """Tester l'algorithme Leiden."""
        clusters = self.clusterer.cluster_leiden()
        self.assertTrue(clusters, "L'algorithme de Leiden n'a détecté aucun cluster.")
        self.assertEqual(sum(len(nodes) for nodes in clusters.values()), self.graph.number_of_nodes())

    def test_cluster_kmeans_embeddings(self):
        """Tester le clustering KMeans sur embeddings de graphe."""
        clusters = self.clusterer.cluster_kmeans_embeddings(n_clusters=4)
        self.assertTrue(clusters, "L'algorithme KMeans n'a détecté aucun cluster.")
        self.assertEqual(sum(len(nodes) for nodes in clusters.values()), self.graph.number_of_nodes())




class TestTextClusterer(unittest.TestCase):
    def setUp(self):
        """Créer un jeu de données textuelles pour les tests."""
        self.documents = ["This is a test document.", "Clustering is useful in machine learning.",
                          "Another text example.", "Graph-based clustering methods are effective."]

        # Simuler un indexeur
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self.indexer = type('Indexer', (object,), {
            'tfidf_matrix': self.tfidf_matrix,
            'doc_term_matrix': self.tfidf_matrix,
            'document_ids': [f"doc_{i}" for i in range(len(self.documents))],
            'feature_names': self.vectorizer.get_feature_names_out()
        })()

        self.clusterer = TextClusterer(self.indexer)

    def test_cluster_kmeans(self):
        """Tester KMeans sur les documents."""
        clusters = self.clusterer.cluster_kmeans(n_clusters=2)
        self.assertTrue(clusters, "KMeans n'a détecté aucun cluster.")
        self.assertEqual(sum(len(docs) for docs in clusters.values()), len(self.documents))

    def test_cluster_dbscan(self):
        """Tester DBSCAN sur les documents."""
        clusters = self.clusterer.cluster_dbscan(eps=0.5, min_samples=1)
        self.assertTrue(clusters, "DBSCAN n'a détecté aucun cluster.")
        self.assertEqual(sum(len(docs) for docs in clusters.values()), len(self.documents))

    def test_cluster_agglomerative(self):
        """Tester le clustering hiérarchique agglomératif."""
        clusters = self.clusterer.cluster_agglomerative(n_clusters=2)
        self.assertTrue(clusters, "L'algorithme Agglomératif n'a détecté aucun cluster.")
        self.assertEqual(sum(len(docs) for docs in clusters.values()), len(self.documents))


class TestLabelGenerator(unittest.TestCase):
    def setUp(self):
        """Créer des documents d'exemple pour tester l'étiquetage."""
        self.documents = [
            {"id": "doc_1", "title": "Graph clustering", "abstract": "Graph-based methods are effective."},
            {"id": "doc_2", "title": "Text analysis",
             "abstract": "Natural Language Processing is useful for clustering."}
        ]

        # Simuler un indexeur avec des termes
        self.indexer = type('Indexer', (object,), {
            'document_ids': ["doc_1", "doc_2"],
            'tfidf_matrix': np.array([[0.2, 0.8], [0.5, 0.5]]),
            'doc_term_matrix': np.array([[1, 3], [2, 1]]),
            'feature_names': ["graph", "clustering"]
        })()

        self.label_generator = LabelGenerator(self.indexer, self.documents)

    def test_get_top_terms(self):
        """Tester l'extraction des termes les plus fréquents."""
        top_terms = self.label_generator.get_top_terms(["doc_1", "doc_2"])
        self.assertTrue(top_terms, "Aucun terme extrait.")
        self.assertTrue(len(top_terms) > 0)




if __name__ == '__main__':
    unittest.main()
