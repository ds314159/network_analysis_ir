

from sentence_transformers import SentenceTransformer


class QueryProcessor:
    def __init__(self, indexer):
        self.indexer = indexer

    def process_query(self, query_text, use_tfidf=True):
        """Traite une requête en texte brut et la transforme en vecteur"""
        # Utiliser le bon vectoriseur selon le paramètre
        vectorizer = self.indexer.tfidf_vectorizer if use_tfidf else self.indexer.vectorizer

        # Transformer la requête en vecteur
        query_vector = vectorizer.transform([query_text])

        return query_vector.toarray().flatten()

    def semantic_embedding(self, query_text, model_name='all-MiniLM-L6-v2'):
        """Obtient une représentation sémantique de la requête avec un modèle transformers"""
        # Charger le modèle Sentence-BERT
        model = SentenceTransformer(model_name)

        # Calculer l'embedding
        embedding = model.encode(query_text)

        return embedding