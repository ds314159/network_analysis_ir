from matplotlib import pyplot as plt
from bertopic import BERTopic
from umap import UMAP
import hdbscan



class BERTopicModel:
    def __init__(self):
        # BERTopic utilise UMAP pour la réduction de dimensionnalité et HDBSCAN pour le clustering
        self.model = BERTopic(
            umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=42),
            hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=100, min_samples=50, prediction_data=True)
        )
        self.topics = None
        self.probs = None

    def fit(self, documents):
        """
        Entraîne BERTopic sur les documents.

        Returns:
            self: Pour permettre le chaînage
        """
        self.topics, self.probs = self.model.fit_transform(documents)
        return self

    def fit_transform(self, documents):
        """
        Entraîne BERTopic sur les documents et retourne les topics.

        Returns:
            topics: Liste des ids de topics assignés à chaque document
            probs: Probabilités d'appartenance de chaque document à son topic
        """
        self.topics, self.probs = self.model.fit_transform(documents)
        return self.topics, self.probs

    def transform(self, documents):
        """
        Retourne la distribution des topics pour de nouveaux documents.

        Returns:
            Un tuple (topics, probs)
        """
        return self.model.transform(documents)

    def get_topics(self, n_words=10):
        """
        Retourne les top words de chaque topic.

        Args:
            n_words: Nombre de mots à retourner par topic

        Returns:
            Un dictionnaire {topic_id: [mots]} avec les mots les plus représentatifs
        """
        # BERTopic.get_topics() ne prend pas de paramètre n_words ou top_n
        # Récupérons d'abord tous les topics
        all_topics = self.model.get_topics()

        # Limitons ensuite le nombre de mots manuellement
        limited_topics = {}
        for topic_id, words in all_topics.items():
            # Limitons à n_words si nécessaire
            limited_topics[topic_id] = words[:n_words] if len(words) > n_words else words

        return limited_topics

    def get_topic_sizes(self):
        """
        Retourne le nombre de documents pour chaque topic.

        Returns:
            DataFrame avec les colonnes Topic et Count
        """
        return self.model.get_topic_freq()

    def get_topic_info(self):
        """
        Retourne des informations détaillées sur chaque topic.

        Returns:
            DataFrame avec des infos sur les topics
        """
        return self.model.get_topic_info()

    def visualize_topics(self):
        """
        Visualise les topics dans un espace 2D.
        """
        return self.model.visualize_topics()

    def visualize_barchart(self, topic_id):
        """
        Crée un graphique à barres pour un topic spécifique.
        """
        return self.model.visualize_barchart(topic=topic_id)

    def show_top_n_topics(self, n=8, num_words=10):
        """
        Affiche et visualise les N topics les plus grands.

        Args:
            n: Nombre de topics à afficher
            num_words: Nombre de mots à afficher par topic
        """
        # Récupérer les topics
        bertopic_topics = self.get_topics(num_words=num_words)

        # Obtenir le nombre total de topics (hors outliers)
        num_topics = len([t for t in bertopic_topics if t != -1])
        print(f"\nNombre total de topics détectés: {num_topics}")

        # Obtenir la taille de chaque topic
        topic_sizes = self.get_topic_sizes()

        # Trier les topics par taille (en excluant le topic -1 qui contient les outliers)
        sorted_topics = topic_sizes[topic_sizes['Topic'] != -1].sort_values('Count', ascending=False)

        # Sélectionner les N topics les plus grands
        top_n_topics = sorted_topics.head(n)

        print(f"\nLes {n} topics les plus grands (par nombre de documents) :")
        print("-" * 60)

        # Afficher uniquement les N topics les plus grands
        for _, row in top_n_topics.iterrows():
            topic_id = row['Topic']
            doc_count = row['Count']
            words = bertopic_topics[topic_id]

            # Extraire seulement les mots (sans les scores) pour l'affichage
            word_list = [word[0] for word in words]
            print(f"Topic {topic_id} ({doc_count} documents): {', '.join(word_list)}")

        print("-" * 60)

        # Visualiser uniquement les N topics les plus grands
        print(f"\nVisualisation des {n} topics les plus grands :")
        for _, row in top_n_topics.iterrows():
            topic_id = row['Topic']
            doc_count = row['Count']
            words = bertopic_topics[topic_id]

            plt.figure(figsize=(10, 6))
            # Extraire les mots et leurs scores
            word_list = [word[0] for word in words]
            score_list = [word[1] for word in words]

            # Créer un graphique à barres horizontales
            plt.barh(range(len(word_list)), score_list, align='center', color='steelblue')
            plt.yticks(range(len(word_list)), word_list)
            plt.xlabel('Score d\'importance')
            plt.title(f"Topic {topic_id} ({doc_count} documents)")
            plt.tight_layout()
            plt.show()

        # Créer un graphique à barres pour comparer les tailles des N topics principaux
        plt.figure(figsize=(12, 6))
        plt.bar(
            top_n_topics['Topic'].astype(str),
            top_n_topics['Count'],
            color='steelblue'
        )
        plt.xlabel('Numéro du topic')
        plt.ylabel('Nombre de documents')
        plt.title(f'Distribution des documents dans les {n} topics les plus grands')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def show_top_n_topics(self, n=8, num_words=10):
        """
        Affiche et visualise les N topics les plus grands.

        Args:
            n: Nombre de topics à afficher
            num_words: Nombre de mots à afficher par topic
        """
        # Récupérer les topics
        bertopic_topics = self.get_topics(n_words=num_words)

        # Obtenir le nombre total de topics (hors outliers)
        num_topics = len([t for t in bertopic_topics if t != -1])
        print(f"\nNombre total de topics détectés: {num_topics}")

        # Obtenir la taille de chaque topic
        topic_sizes = self.get_topic_sizes()

        # Trier les topics par taille (en excluant le topic -1 qui contient les outliers)
        sorted_topics = topic_sizes[topic_sizes['Topic'] != -1].sort_values('Count', ascending=False)

        # Sélectionner les N topics les plus grands
        top_n_topics = sorted_topics.head(n)

        print(f"\nLes {n} topics les plus grands (par nombre de documents) :")
        print("-" * 60)

        # Afficher uniquement les N topics les plus grands
        for _, row in top_n_topics.iterrows():
            topic_id = row['Topic']
            doc_count = row['Count']
            words = bertopic_topics[topic_id]

            # Extraire seulement les mots (sans les scores) pour l'affichage
            word_list = [word[0] for word in words]
            print(f"Topic {topic_id} ({doc_count} documents): {', '.join(word_list)}")

        print("-" * 60)

        # Visualiser uniquement les N topics les plus grands
        print(f"\nVisualisation des {n} topics les plus grands :")
        for _, row in top_n_topics.iterrows():
            topic_id = row['Topic']
            doc_count = row['Count']
            words = bertopic_topics[topic_id]

            plt.figure(figsize=(10, 6))
            # Extraire les mots et leurs scores
            word_list = [word[0] for word in words]
            score_list = [word[1] for word in words]

            # Créer un graphique à barres horizontales
            plt.barh(range(len(word_list)), score_list, align='center', color='steelblue')
            plt.yticks(range(len(word_list)), word_list)
            plt.xlabel('Score d\'importance')
            plt.title(f"Topic {topic_id} ({doc_count} documents)")
            plt.tight_layout()
            plt.show()

        # Créer un graphique à barres pour comparer les tailles des N topics principaux
        plt.figure(figsize=(12, 6))
        plt.bar(
            top_n_topics['Topic'].astype(str),
            top_n_topics['Count'],
            color='steelblue'
        )
        plt.xlabel('Numéro du topic')
        plt.ylabel('Nombre de documents')
        plt.title(f'Distribution des documents dans les {n} topics les plus grands')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
