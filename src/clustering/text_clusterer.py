import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.sparse import issparse
from typing import Dict, List
from collections import defaultdict

class TextClusterer:
    """Classe permettant d'appliquer des méthodes de clustering sur des documents textuels"""

    def __init__(self, indexer):
        """Initialiser la classe avec un indexeur de documents"""
        self.indexer = indexer
        self.model = None
        self.clusters = None
        self.cluster_centers = None

    def cluster_kmeans(self, n_clusters=8, use_tfidf=True, random_state=42) -> Dict[int, List[str]]:
        """Appliquer l'algorithme K-means sur les représentations vectorielles des documents"""
        # Sélectionner la matrice TF-IDF ou la matrice document-terme
        matrix = self.indexer.tfidf_matrix if use_tfidf else self.indexer.doc_term_matrix

        # Initialiser et entraîner le modèle K-means
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = self.model.fit_predict(matrix)

        # Stocker les centres des clusters
        self.cluster_centers = self.model.cluster_centers_

        # Structurer les résultats en dictionnaire de clusters
        self.clusters = defaultdict(list)
        for i, label in enumerate(labels):
            self.clusters[label].append(self.indexer.document_ids[i])

        return self.clusters

    def get_closest_documents_to_center(self, cluster_id: int, top_n=5) -> List[str]:
        """Retourner les documents les plus proches du centre d'un cluster"""
        # Vérifier que le modèle utilisé supporte la notion de centre
        if not hasattr(self.model, 'cluster_centers_'):
            raise ValueError("Le modèle utilisé ne supporte pas la notion de centre (ex: DBSCAN).")

        # Vérifier que le cluster demandé existe
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} non trouvé.")

        # Récupérer les documents du cluster
        doc_ids = self.clusters[cluster_id]

        # Calculer la distance de chaque document au centre du cluster
        distances = [
            (doc_id, np.linalg.norm(
                self.indexer.tfidf_matrix[self.indexer.document_ids.index(doc_id)].toarray()
                - self.cluster_centers[cluster_id])
            ) for doc_id in doc_ids
        ]

        # Trier les documents par distance croissante et retourner les plus proches
        return [doc_id for doc_id, _ in sorted(distances, key=lambda x: x[1])[:top_n]]

    def cluster_agglomerative(self, n_clusters=8, linkage='ward', use_tfidf=True) -> Dict[int, List[str]]:
        """Appliquer le clustering hiérarchique agglomératif."""
        matrix = self.indexer.tfidf_matrix if use_tfidf else self.indexer.doc_term_matrix

        # Convertir en dense si la matrice est creuse
        matrix_dense = matrix.toarray() if issparse(matrix) else matrix

        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(matrix_dense)

        self.clusters = {}
        for i, label in enumerate(labels):
            self.clusters.setdefault(label, []).append(self.indexer.document_ids[i])

        return self.clusters

    def cluster_dbscan(self, eps=0.5, min_samples=5, use_tfidf=True) -> Dict[int, List[str]]:
        """Appliquer DBSCAN sur les représentations vectorielles des documents."""
        matrix = self.indexer.tfidf_matrix if use_tfidf else self.indexer.doc_term_matrix

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(matrix)

        self.clusters = {}
        for i, label in enumerate(labels):
            self.clusters.setdefault(label, []).append(self.indexer.document_ids[i])

        return self.clusters

