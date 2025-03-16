import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class GraphStats:
    def __init__(self, graph):
        self.graph = graph

    def compute_basic_stats(self):
        """Calcule des statistiques de base sur le graphe"""
        is_directed = isinstance(self.graph, nx.DiGraph)

        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_strongly_connected(self.graph) if is_directed else nx.is_connected(self.graph),
            'num_connected_components': nx.number_strongly_connected_components(self.graph) if is_directed
                                        else nx.number_connected_components(self.graph),
            'average_clustering': nx.average_clustering(self.graph.to_undirected()) if is_directed
                                  else nx.average_clustering(self.graph),
            'diameter': self._get_diameter()
        }
        return stats

    def compute_centrality_measures(self):
        """Calcule différentes mesures de centralité"""
        try:
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=500, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            eigenvector = None  # En cas de non-convergence, renvoyer None

        centrality = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'closeness': nx.closeness_centrality(self.graph),
            'eigenvector': eigenvector
        }
        return centrality

    def get_degree_distribution(self):
        """Calcule la distribution des degrés"""
        degrees = dict(self.graph.degree())
        degree_counts = {}
        for degree in degrees.values():
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        return degree_counts

    def _get_diameter(self):
        """Calcule le diamètre du graphe (ou d'une composante si non connexe)"""
        try:
            if isinstance(self.graph, nx.DiGraph):
                if nx.is_strongly_connected(self.graph):
                    return nx.diameter(self.graph)
                else:
                    # Trouver la plus grande composante fortement connexe
                    components = list(nx.strongly_connected_components(self.graph))
            else:
                if nx.is_connected(self.graph):
                    return nx.diameter(self.graph)
                else:
                    components = list(nx.connected_components(self.graph))

            # Calcul du diamètre de la plus grande composante
            largest_component = max(components, key=len)
            subgraph = self.graph.subgraph(largest_component)
            return nx.diameter(subgraph)

        except nx.NetworkXError:
            return None  # Cas des graphes totalement déconnectés