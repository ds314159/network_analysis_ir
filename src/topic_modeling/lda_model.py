import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class LDATopicModel:
    def __init__(self, num_topics=10, max_iter=10):
        """
        Initialise un modèle de topic modeling basé sur LDA.

        Args:
            num_topics: Nombre de topics à extraire
            max_iter: Nombre maximum d'itérations pour la convergence
        """
        self.num_topics = num_topics
        self.max_iter = max_iter
        # LDA modélise les documents comme des mélanges de topics
        # où chaque topic est une distribution de probabilité sur les mots
        self.model = LatentDirichletAllocation(n_components=num_topics, max_iter=max_iter, random_state=42)
        # CountVectorizer transforme les documents en matrices terme-fréquence
        self.vectorizer = CountVectorizer(stop_words='english')

    def fit(self, documents):
        """
        Entraîne le modèle LDA sur un corpus de documents.

        Processus:
        1. Transformation des documents en matrice terme-document
        2. Estimation des distributions de topics via LDA

        Args:
            documents: Liste de documents textuels
        """
        self.doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.model.fit(self.doc_term_matrix)

    def get_topics(self, num_words=10):
        """
        Retourne les topics avec leurs mots-clés.

        Pour chaque topic, retourne les mots ayant les coefficients
        les plus élevés dans la distribution du topic.

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

        Transforme de nouveaux documents en matrice terme-document
        puis calcule leur distribution sur les topics appris.

        Args:
            documents: Liste de documents textuels

        Returns:
            Une matrice où chaque ligne est un document et chaque colonne
            est la proportion du topic dans ce document
        """
        doc_term_matrix = self.vectorizer.transform(documents)
        return self.model.transform(doc_term_matrix)