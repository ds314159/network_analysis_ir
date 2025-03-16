import networkx as nx
import numpy as np
import json
from typing import Dict, List, Any, Optional
#import community as community_louvain
from sklearn.cluster import SpectralClustering, KMeans
from cdlib import algorithms  # Bibliothèque pour des algorithmes avancés de clustering de graphes
from networkx.algorithms.community import label_propagation
from community import community_louvain

class GraphClusterer:
    """Classe permettant d'appliquer plusieurs méthodes de clustering sur un graphe."""

    def __init__(self, graph: nx.Graph):
        """Initialiser le clusteriseur de graphe avec un objet NetworkX."""
        self.graph = graph
        self.clusters = None

    def cluster_louvain(self) -> Dict[int, List[Any]]:
        """Appliquer l'algorithme de Louvain pour détecter les communautés."""
        communities = community_louvain.best_partition(self.graph)
        self.clusters = {}

        # Organiser les nœuds par cluster
        for node, cluster_id in communities.items():
            self.clusters.setdefault(cluster_id, []).append(node)

        return self.clusters

    def cluster_spectral(self, n_clusters=8) -> Dict[int, List[Any]]:
        """Appliquer le clustering spectral basé sur la matrice d'adjacence."""
        if self.graph.number_of_nodes() < n_clusters:
            raise ValueError("Le nombre de clusters doit être inférieur au nombre de nœuds du graphe.")

        # Convertir le graphe en matrice d'adjacence
        adj_matrix = nx.to_numpy_array(self.graph)

        # Appliquer Spectral Clustering
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize', random_state=42)
        labels = sc.fit_predict(adj_matrix)

        self.clusters = {}
        nodes = list(self.graph.nodes())

        for i, label in enumerate(labels):
            self.clusters.setdefault(label, []).append(nodes[i])

        return self.clusters

    def cluster_label_propagation(self) -> Dict[int, List[Any]]:
        """Appliquer Label Propagation, un algorithme de clustering non supervisé basé sur la diffusion d'étiquettes."""
        communities = list(label_propagation.asyn_lpa_communities(self.graph))

        self.clusters = {i: list(community) for i, community in enumerate(communities)}

        return self.clusters

    def cluster_leiden(self) -> Dict[int, List[Any]]:
        """Appliquer l'algorithme de Leiden pour détecter les communautés."""
        # Leiden nécessite que le graphe soit pondéré
        leiden_result = algorithms.leiden(self.graph, weights="weight")
        communities = leiden_result.communities

        self.clusters = {i: list(community) for i, community in enumerate(communities)}

        return self.clusters

    def cluster_kmeans_embeddings(self, n_clusters=8) -> Dict[int, List[Any]]:
        """Appliquer KMeans sur des embeddings de graphes pour identifier les clusters."""
        if self.graph.number_of_nodes() < n_clusters:
            raise ValueError("Le nombre de clusters doit être inférieur au nombre de nœuds du graphe.")

        # Convertir en matrice d'adjacence normalisée
        adj_matrix = nx.to_numpy_array(self.graph)

        # Appliquer KMeans sur les embeddings issus de la décomposition spectrale
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(adj_matrix)

        self.clusters = {}
        nodes = list(self.graph.nodes())

        for i, label in enumerate(labels):
            self.clusters.setdefault(label, []).append(nodes[i])

        return self.clusters

    def get_subgraph_for_cluster(self, cluster_id: int) -> nx.Graph:
        """Retourner un sous-graphe correspondant à un cluster spécifique."""
        if self.clusters is None:
            raise ValueError("Exécutez d'abord une méthode de clustering.")

        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} non trouvé.")

        return self.graph.subgraph(self.clusters[cluster_id])

    def save_clusters(self, file_path: str):
        """Enregistre les résultats du clustering sous forme de fichier JSON avec des clés en chaîne."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in self.clusters.items()}, f, indent=4)
        print(f"Clusters enregistrés dans {file_path}")

    def load_clusters(self, file_path: str):
        """Charge les clusters depuis un fichier JSON et reconvertit les clés en entiers."""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.clusters = {int(k): v for k, v in json.load(f).items()}
        print(f"Clusters chargés depuis {file_path}")

