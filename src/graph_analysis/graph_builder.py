import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import pandas as pd



class GraphBuilder:
    def __init__(self, data):
        self.data = data

    def build_citation_graph(self):
        """Construit un graphe de citation orienté (document → document)"""
        G = nx.DiGraph()

        # Ajouter les nœuds (documents)
        doc_ids = set()  # Stocker les IDs pour vérification rapide des références
        for doc in self.data:
            if 'id' in doc:
                doc_ids.add(doc['id'])  # Ajouter au set pour vérification ultérieure
                G.add_node(doc['id'],
                           title=doc.get('title', ''),
                           year=doc.get('year', None),
                           class_id=doc.get('class', None))

        # Ajouter les arêtes (citations)
        for doc in self.data:
            if 'id' in doc and 'references' in doc:
                doc_id = doc['id']
                for ref_id in doc['references']:
                    if ref_id in doc_ids:  # Vérifier que le document cité existe
                        G.add_edge(doc_id, ref_id)


        print(f"Graphe construit avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes.")
        return G

    def build_coauthorship_graph(self):
        """Construit un graphe de co-auteurs (auteur -- auteur)"""
        G = nx.Graph()

        # Créer un dictionnaire pour stocker les documents par auteur
        author_docs = {}
        for doc in self.data:
            if 'authors' in doc:
                for author in doc['authors']:
                    if author not in author_docs:
                        author_docs[author] = []
                    author_docs[author].append(doc['id'])

        # Ajouter les nœuds (auteurs)
        for author, docs in author_docs.items():
            G.add_node(author, num_papers=len(docs))

        # Ajouter les arêtes (co-autorat)
        for doc in self.data:
            if 'authors' in doc and len(doc['authors']) > 1:
                authors = doc['authors']
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        if G.has_edge(authors[i], authors[j]):
                            # Incrémenter le poids de l'arête si elle existe déjà
                            G[authors[i]][authors[j]]['weight'] += 1
                        else:
                            G.add_edge(authors[i], authors[j], weight=1)

        return G

    def build_document_author_graph(self):
        """Construit un graphe biparti document-auteur"""
        G = nx.Graph()

        # Ajouter les nœuds (documents et auteurs)
        for doc in self.data:
            if 'id' in doc:
                G.add_node(doc['id'], type='document',
                           title=doc.get('title', ''),
                           class_id=doc.get('class', None))

                if 'authors' in doc:
                    for author in doc['authors']:
                        if not G.has_node(author):
                            G.add_node(author, type='author')
                        G.add_edge(doc['id'], author)

        return G

    def build_venue_document_graph(self):
        """Construit un graphe biparti venue-document"""
        G = nx.Graph()

        # Ajouter les nœuds (venues et documents)
        for doc in self.data:
            if 'id' in doc and 'venue' in doc:
                venue = doc['venue']

                if not G.has_node(doc['id']):
                    G.add_node(doc['id'], type='document',
                               title=doc.get('title', ''),
                               class_id=doc.get('class', None))

                if not G.has_node(venue):
                    G.add_node(venue, type='venue')

                G.add_edge(doc['id'], venue)

        return G