import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from nltk.tokenize import word_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Vérifier et télécharger les stopwords si nécessaire
try:
    _ = stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('punkt')


class LabelGenerator:
    def __init__(self, indexer, documents: List[Dict[str, Any]], id_field='id'):
        self.indexer = indexer
        self.documents = {doc[id_field]: doc for doc in documents if id_field in doc}

    def get_top_terms(self, cluster_doc_ids: List[str], top_n=10) -> List[Tuple[str, int]]:
        """Retourne les termes les plus fréquents dans un cluster."""
        doc_indices = [self.indexer.document_ids.index(doc_id) for doc_id in cluster_doc_ids if
                       doc_id in self.indexer.document_ids]

        if not doc_indices:
            return []

        cluster_matrix = self.indexer.doc_term_matrix[doc_indices]
        term_freqs = np.asarray(cluster_matrix.sum(axis=0)).flatten()
        term_indices = term_freqs.argsort()[-top_n:][::-1]

        return [(self.indexer.feature_names[i], term_freqs[i]) for i in term_indices]

    def get_discriminative_terms(self, cluster_doc_ids, num_terms=10):
        """Retourne les termes les plus discriminants pour un cluster"""
        # Obtenir les indices des documents dans le cluster
        doc_indices = [self.indexer.document_ids.index(doc_id)
                       for doc_id in cluster_doc_ids
                       if doc_id in self.indexer.document_ids]

        other_indices = [i for i in range(len(self.indexer.document_ids)) if i not in doc_indices]

        if not doc_indices or not other_indices:
            return []

        # Extraire les sous-matrices correspondant aux documents du cluster et aux autres
        cluster_matrix = self.indexer.tfidf_matrix[doc_indices].toarray().mean(axis=0)
        other_matrix = self.indexer.tfidf_matrix[other_indices].toarray().mean(axis=0)

        # Calculer le score de discrimination (différence entre moyenne dans le cluster et ailleurs)
        discrimination_scores = cluster_matrix - other_matrix

        # Trier les termes par score décroissant
        term_indices = discrimination_scores.argsort()[-num_terms:][::-1]

        # Retourner les termes et leurs scores
        terms = [self.indexer.feature_names[i] for i in term_indices]
        scores = [discrimination_scores[i] for i in term_indices]

        return list(zip(terms, scores))

    def extract_collocations(self, cluster_doc_ids: List[str], min_count=5) -> List[str]:
        """Extrait les expressions significatives d'un cluster."""
        texts = [self.documents[doc_id].get('abstract', '') for doc_id in cluster_doc_ids if doc_id in self.documents]
        combined_text = ' '.join(texts)

        tokens = word_tokenize(combined_text.lower())
        filtered_tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english')]

        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(filtered_tokens)
        finder.apply_freq_filter(min_count)

        return [' '.join(bigram) for bigram in finder.nbest(bigram_measures.pmi, 10)]