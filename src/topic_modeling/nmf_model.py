from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class NMFTopicModel:
    def __init__(self, num_topics=10):
        """
        Initialise un modèle de topic modeling basé sur NMF.

        NMF (Non-negative Matrix Factorization) est une technique de
        factorisation de matrices qui décompose une matrice en deux
        matrices de facteurs non-négatifs.

        Args:
            num_topics: Nombre de topics à extraire
        """
        self.num_topics = num_topics
        # NMF décompose la matrice document-terme en deux matrices:
        # - matrice document-topic (coefficient matrix)
        # - matrice topic-terme (composantes)
        self.model = NMF(n_components=num_topics, random_state=42)
        # CountVectorizer pour la représentation bag-of-words
        self.vectorizer = CountVectorizer(stop_words='english')

    def fit(self, documents):
        """
        Entraîne le modèle NMF.

        Processus:
        1. Transformation des documents en matrice terme-document
        2. Factorisation de cette matrice via NMF pour extraire les topics

        Args:
            documents: Liste de documents textuels
        """
        self.doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.model.fit(self.doc_term_matrix)

    def get_topics(self, num_words=10):
        """
        Retourne les mots-clés des topics.

        Dans NMF, les composantes correspondent aux poids des termes
        dans chaque topic. Les mots avec les poids les plus élevés
        sont considérés comme les plus représentatifs du topic.

        Args:
            num_words: Nombre de mots à retourner par topic

        Returns:
            Un dictionnaire {topic_name: [mots]} avec les mots représentatifs
        """
        words = np.array(self.vectorizer.get_feature_names_out())
        topics = {}
        for topic_idx, topic in enumerate(self.model.components_):
            # Tri des mots par importance décroissante dans le topic
            top_words = words[np.argsort(topic)][:-num_words - 1:-1]
            topics[f"Topic {topic_idx}"] = top_words.tolist()
        return topics

    def transform(self, documents):
        """
        Retourne la distribution des topics pour chaque document.

        Projette de nouveaux documents dans l'espace des topics
        appris lors de l'entraînement.

        Args:
            documents: Liste de documents textuels

        Returns:
            Une matrice où chaque ligne est un document et chaque colonne
            représente le poids du topic dans ce document
        """
        doc_term_matrix = self.vectorizer.transform(documents)
        return self.model.transform(doc_term_matrix)