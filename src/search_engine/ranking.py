import numpy as np
from src.search_engine.query_processor import QueryProcessor
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
login("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")

from scipy.sparse import issparse
from sklearn.preprocessing import normalize


class RankingSystem:
    def __init__(self, indexer, documents, id_field='id'):
        self.indexer = indexer
        self.documents = {doc[id_field]: doc for doc in documents if id_field in doc}

    import numpy as np
    from scipy.sparse import issparse
    from sklearn.preprocessing import normalize

    def search(self, query_vector, top_n=10, similarity_measure='cosine', use_tfidf=True):
        """Recherche les documents les plus similaires à la requête"""
        # Choisir la matrice appropriée (TF-IDF ou BoW)
        matrix = self.indexer.tfidf_matrix if use_tfidf else self.indexer.doc_term_matrix

        if similarity_measure == 'cosine':
            # Normaliser la matrice et le vecteur requête pour la similarité cosinus
            matrix = normalize(matrix, norm='l2', axis=1)  # Normalisation ligne par ligne
            query_vector = normalize(query_vector.reshape(1, -1), norm='l2')

            # Calcul de la similarité cosinus
            similarity_scores = matrix.dot(query_vector.T).flatten()

        elif similarity_measure == 'euclidean':
            # Calcul de la distance euclidienne (convertie en similarité)
            similarity_scores = np.array([
                1 / (1 + np.linalg.norm(matrix[i].toarray().flatten() - query_vector))
                for i in range(matrix.shape[0])
            ])
        else:
            raise ValueError(f"Mesure de similarité non supportée : {similarity_measure}")

        # Trier les documents par score de similarité décroissant
        doc_indices = similarity_scores.argsort()[-top_n:][::-1]

        # Préparer les résultats avec les scores et les informations des documents
        results = []
        for idx in doc_indices:
            doc_id = self.indexer.document_ids[idx]
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append({
                    'id': doc_id,
                    'score': float(similarity_scores[idx]),
                    'title': doc.get('title', ''),
                    'abstract': doc.get('abstract', ''),
                    'authors': doc.get('authors', []),
                    'year': doc.get('year', None),
                    'venue': doc.get('venue', ''),
                    'class': doc.get('class', None)
                })

        return results

    def search_semantic(self, query_embedding, document_embeddings, document_ids, top_n=10):
        """Recherche basée sur les embeddings sémantiques"""
        # Calculer la similarité cosinus entre la requête et tous les documents
        similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]

        # Trier les documents par score de similarité décroissant
        doc_indices = similarity_scores.argsort()[-top_n:][::-1]

        # Préparer les résultats avec les scores et les informations des documents
        results = []
        for idx in doc_indices:
            doc_id = document_ids[idx]
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append({
                    'id': doc_id,
                    'score': float(similarity_scores[idx]),
                    'title': doc.get('title', ''),
                    'abstract': doc.get('abstract', ''),
                    'authors': doc.get('authors', []),
                    'year': doc.get('year', None),
                    'venue': doc.get('venue', ''),
                    'class': doc.get('class', None)
                })

        return results

    def combine_ranking_methods(self, query_text, top_n=10, weights={'tfidf': 0.3, 'semantic': 0.7}):
        """Combine différentes méthodes de classement pour améliorer les résultats"""
        # Traiter la requête avec la méthode TF-IDF
        query_processor = QueryProcessor(self.indexer)
        query_vector_tfidf = query_processor.process_query(query_text, use_tfidf=True)
        results_tfidf = self.search(query_vector_tfidf, top_n=top_n * 2)  # Obtenir plus de résultats pour la fusion

        # Traiter la requête avec l'embedding sémantique
        query_embedding = query_processor.semantic_embedding(query_text)

        # Calculer les embeddings sémantiques pour tous les documents (simplification - en pratique, cela pourrait être prétraité)
        document_embeddings = []
        document_ids = []
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        for doc_id in self.indexer.document_ids:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                text = doc.get('abstract', '') or doc.get('title', '')
                if text:
                    document_embeddings.append(model.encode(text))
                    document_ids.append(doc_id)

        # Recherche sémantique
        results_semantic = self.search_semantic(query_embedding, document_embeddings, document_ids, top_n=top_n * 2)

        # Fusionner les résultats en utilisant les poids
        merged_scores = {}
        for result in results_tfidf:
            doc_id = result['id']
            merged_scores[doc_id] = weights['tfidf'] * result['score']

        for result in results_semantic:
            doc_id = result['id']
            if doc_id in merged_scores:
                merged_scores[doc_id] += weights['semantic'] * result['score']
            else:
                merged_scores[doc_id] = weights['semantic'] * result['score']

        # Trier les documents par score fusionné
        ranked_doc_ids = sorted(merged_scores.keys(), key=lambda x: merged_scores[x], reverse=True)[:top_n]

        # Préparer les résultats finaux
        results = []
        for doc_id in ranked_doc_ids:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append({
                    'id': doc_id,
                    'score': merged_scores[doc_id],
                    'title': doc.get('title', ''),
                    'abstract': doc.get('abstract', ''),
                    'authors': doc.get('authors', []),
                    'year': doc.get('year', None),
                    'venue': doc.get('venue', ''),
                    'class': doc.get('class', None)
                })

        return results

    def search_with_text_query(self, query_text, top_n=10, use_tfidf=True, similarity_measure='cosine'):
        """Recherche avec une requête textuelle directement"""
        from src.search_engine.query_processor import QueryProcessor

        # Créer le processeur de requêtes
        query_processor = QueryProcessor(self.indexer)

        # Traiter la requête
        query_vector = query_processor.process_query(query_text, use_tfidf=use_tfidf)

        # Rechercher les documents pertinents
        return self.search(query_vector, top_n=top_n, similarity_measure=similarity_measure, use_tfidf=use_tfidf)

    def compare_search_methods(self, query_text, top_n=3):
        """Compare différentes méthodes de recherche et retourne les résultats"""
        results = {
            'tf': self.search_with_text_query(query_text, top_n=top_n, use_tfidf=False),
            'tfidf': self.search_with_text_query(query_text, top_n=top_n, use_tfidf=True),
            'cosine': self.search_with_text_query(query_text, top_n=top_n, similarity_measure='cosine'),
            'euclidean': self.search_with_text_query(query_text, top_n=top_n, similarity_measure='euclidean')
        }

        # Calculer les chevauchements
        tf_tfidf_common = set([r['id'] for r in results['tf']]) & set([r['id'] for r in results['tfidf']])
        cosine_euclidean_common = set([r['id'] for r in results['cosine']]) & set(
            [r['id'] for r in results['euclidean']])

        comparison = {
            'results': results,
            'overlap': {
                'tf_tfidf': len(tf_tfidf_common),
                'cosine_euclidean': len(cosine_euclidean_common)
            }
        }

        return comparison

    def analyze_query_results(self, query, top_n=3, use_tfidf=True, similarity_measure='cosine'):
        """Analyse les résultats d'une requête en termes de classes dominantes"""
        # Effectuer la recherche
        results = self.search_with_text_query(
            query,
            top_n=top_n,
            use_tfidf=use_tfidf,
            similarity_measure=similarity_measure
        )

        # Analyser les classes des résultats
        from collections import Counter
        class_counts = Counter([result['class'] for result in results if result.get('class') is not None])
        dominant_class = class_counts.most_common(1)[0][0] if class_counts else None

        # Calculer le score moyen
        avg_score = sum(result['score'] for result in results) / len(results) if results else 0

        analysis = {
            'dominant_class': dominant_class,
            'class_distribution': dict(class_counts),
            'avg_score': avg_score,
            'results': results
        }

        return analysis