from src.topic_modeling.lda_model import LDATopicModel
from src.topic_modeling.nmf_model import NMFTopicModel
from src.topic_modeling.bert_model import BERTopicModel


class TopicExtractor:
    def __init__(self, model_type='lda', num_topics=10):
        """
        Classe façade qui unifie l'interface des différents modèles de topic modeling.

        Permet d'utiliser LDA, NMF ou BERTopic avec la même interface, facilitant
        l'expérimentation avec différentes approches.

        Args:
            model_type: Type de modèle à utiliser ('lda', 'nmf', ou 'bertopic')
            num_topics: Nombre de topics à extraire (ignoré pour BERTopic qui détermine
                        automatiquement le nombre optimal)
        """
        if model_type == 'lda':
            self.model = LDATopicModel(num_topics=num_topics)
        elif model_type == 'nmf':
            self.model = NMFTopicModel(num_topics=num_topics)
        elif model_type == 'bertopic':
            self.model = BERTopicModel()
        else:
            raise ValueError("Modèle non supporté")

    def fit(self, documents):
        """
        Entraîne le modèle de topic sur le corpus de documents.

        Cette méthode délègue l'entraînement au modèle spécifique choisi,
        en maintenant une interface unifiée.

        Args:
            documents: Liste de documents textuels à analyser
        """
        self.model.fit(documents)

    def get_topics(self, num_words=10):
        """
        Retourne les topics identifiés avec leurs mots-clés.

        Args:
            num_words: Nombre de mots à retourner par topic

        Returns:
            Structure dépendant du modèle sous-jacent, généralement un dictionnaire
            de topics avec leurs mots-clés associés
        """
        return self.model.get_topics(num_words)

    def transform(self, documents):
        """
        Retourne la distribution des topics pour les documents fournis.

        Utile pour classifier de nouveaux documents après l'entraînement
        ou pour analyser la distribution des topics sur le corpus.

        Args:
            documents: Liste de documents textuels

        Returns:
            Format dépendant du modèle sous-jacent, généralement une matrice
            document-topic indiquant la proportion de chaque topic dans chaque document
        """
        return self.model.transform(documents)