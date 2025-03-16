import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pyvis.network import Network
import numpy as np
from typing import Dict, List, Any, Optional, Union
import plotly.graph_objects as go
import ipysigma
import community as community_louvain
import matplotlib.cm as cm

class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph

    def visualize_graph(self, layout='spring', node_color_attribute=None, node_size_attribute=None,
                        edge_width_attribute=None, title='Graph Visualization'):
        """Visualise le graphe avec différentes options de mise en page et d'attributs"""
        plt.figure(figsize=(12, 10))

        # Choisir la mise en page
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, seed=42)

        # Configurer les couleurs des nœuds
        node_colors = self._get_node_colors(node_color_attribute) if node_color_attribute else 'skyblue'

        # Configurer les tailles des nœuds
        node_sizes = self._get_node_sizes(node_size_attribute) if node_size_attribute else 300

        # Configurer les largeurs des arêtes
        edge_widths = self._get_edge_widths(edge_width_attribute) if edge_width_attribute else 1.0

        # Dessiner le graphe
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(self.graph, pos, width=edge_widths, alpha=0.7)
        nx.draw_networkx_labels(self.graph, pos, font_size=8)

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_communities(self, communities, layout='spring', title='Community Structure'):
        """Visualise la structure communautaire du graphe"""
        plt.figure(figsize=(12, 10))

        # Attribuer une couleur à chaque communauté
        color_map = {}
        for i, community in enumerate(communities):
            for node in community:
                color_map[node] = i

        node_colors = [color_map.get(node, -1) for node in self.graph.nodes()]

        # Choisir la mise en page
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, seed=42)

        # Dessiner le graphe
        nx.draw_networkx(self.graph, pos, node_color=node_colors, cmap=plt.cm.rainbow,
                         node_size=300, edge_color='gray', width=0.5, with_labels=False)

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _get_node_colors(self, attribute):
        """Détermine les couleurs des nœuds selon un attribut"""
        if attribute not in next(iter(self.graph.nodes(data=True)))[1]:
            return 'skyblue'  # Couleur par défaut si attribut non trouvé

        values = [data.get(attribute, 0) for node, data in self.graph.nodes(data=True)]

        # Si l'attribut est catégoriel
        if isinstance(values[0], (str, bool)) or (isinstance(values[0], (int, float)) and len(set(values)) < 10):
            # Créer une palette de couleurs pour les valeurs distinctes
            unique_values = list(set(values))
            color_map = {val: plt.cm.tab10(i / len(unique_values)) for i, val in enumerate(unique_values)}
            return [color_map[val] for val in values]

        # Si l'attribut est numérique
        return values

    def _get_node_sizes(self, attribute):
        """Détermine les tailles des nœuds selon un attribut"""
        if attribute not in next(iter(self.graph.nodes(data=True)))[1]:
            return 300  # Taille par défaut si attribut non trouvé

        values = [data.get(attribute, 1) for node, data in self.graph.nodes(data=True)]

        # Normaliser les valeurs pour les tailles des nœuds
        if all(isinstance(v, (int, float)) for v in values):
            min_val, max_val = min(values), max(values)
            if min_val == max_val:
                return 300
            else:
                return [100 + 900 * (v - min_val) / (max_val - min_val) for v in values]

        return 300

    def _get_edge_widths(self, attribute):
        """Détermine les largeurs des arêtes selon un attribut"""
        if not nx.get_edge_attributes(self.graph, attribute):
            return 1.0  # Largeur par défaut si attribut non trouvé

        values = [data.get(attribute, 1) for u, v, data in self.graph.edges(data=True)]

        # Normaliser les valeurs pour les largeurs des arêtes
        if all(isinstance(v, (int, float)) for v in values):
            min_val, max_val = min(values), max(values)
            if min_val == max_val:
                return 1.0
            else:
                return [0.5 + 4.5 * (v - min_val) / (max_val - min_val) for v in values]

        return 1.0

    def visualize_graph_tsne(graph, node_to_cluster, title="Clusters visualization with t-SNE", perplexity=6):
        # Obtenir la matrice d'adjacence
        adj_matrix = nx.to_numpy_array(graph)

        # Appliquer t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        positions = tsne.fit_transform(adj_matrix)

        # Préparer les couleurs
        unique_clusters = sorted(set(node_to_cluster.values()))
        color_map = plt.cm.get_cmap('tab10', len(unique_clusters))

        # Créer une figure plus grande
        plt.figure(figsize=(12, 10))

        # Dessiner les nœuds avec des couleurs basées sur les clusters
        for i, node in enumerate(graph.nodes()):
            cluster = node_to_cluster.get(node, -1)
            color_idx = unique_clusters.index(cluster) if cluster in unique_clusters else 0
            plt.scatter(positions[i, 0], positions[i, 1], c=[color_map(color_idx)], s=50, alpha=0.8)

        # Dessiner les arêtes en gris clair
        for i, j in graph.edges():
            i_idx = list(graph.nodes()).index(i)
            j_idx = list(graph.nodes()).index(j)
            plt.plot([positions[i_idx, 0], positions[j_idx, 0]],
                     [positions[i_idx, 1], positions[j_idx, 1]],
                     'lightgray', alpha=0.2, linewidth=0.5)

        # Ajouter titre et légende
        plt.title(title, fontsize=16)

        # Créer une légende pour les clusters
        for i, cluster_id in enumerate(unique_clusters):
            plt.scatter([], [], c=[color_map(i)], label=f'Cluster {cluster_id}')

        plt.legend(loc='best')
        plt.axis('off')
        plt.tight_layout()
        plt.show()