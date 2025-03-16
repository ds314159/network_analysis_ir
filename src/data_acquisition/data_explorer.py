import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

class DataExplorer:
    def __init__(self, data):
        self.data = data

    def get_basic_stats(self):
        """Calcule des statistiques de base sur les données"""
        stats = {
            'num_documents': len(self.data),
            'num_authors': self._count_unique_authors(),
            'avg_text_length': self._calculate_avg_text_length(),
            'class_distribution': self._get_class_distribution(),
            'temporal_distribution': self._get_temporal_distribution() if 'year' in self.data[0] else None
        }
        return stats

    def visualize_class_distribution(self):
        """Visualise la distribution des classes"""
        class_dist = self._get_class_distribution()
        plt.figure(figsize=(10, 6))
        plt.bar(class_dist.keys(), class_dist.values())
        plt.title('Distribution des classes')
        plt.xlabel('Classe')
        plt.ylabel('Nombre de documents')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _count_unique_authors(self):
        """Compte le nombre d'auteurs uniques"""
        all_authors = set()
        for doc in self.data:
            if 'authors' in doc:
                for author in doc['authors']:
                    all_authors.add(author)
        return len(all_authors)

    def _calculate_avg_text_length(self):
        """Calcule la longueur moyenne des textes"""
        lengths = []
        for doc in self.data:
            if 'abstract' in doc and doc['abstract']:
                lengths.append(len(doc['abstract'].split()))
            elif 'title' in doc:
                lengths.append(len(doc['title'].split()))
        return sum(lengths) / len(lengths) if lengths else 0

    def _get_class_distribution(self):
        """Calcule la distribution des classes"""
        class_dist = {}
        for doc in self.data:
            if 'class' in doc:
                cls = doc['class']
                class_dist[cls] = class_dist.get(cls, 0) + 1
        return class_dist

    def _get_temporal_distribution(self):
        """Calcule la distribution temporelle des documents"""
        temp_dist = {}
        for doc in self.data:
            if 'year' in doc:
                year = doc['year']
                temp_dist[year] = temp_dist.get(year, 0) + 1
        return temp_dist