import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import issparse, csr_matrix
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
import joblib
from collections import Counter

class Indexer:
    def __init__(self, config):
        self.config = config
        self.vectorizer = None
        self.tfidf_vectorizer = None
        self.doc_term_matrix = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.document_ids = None

    def build_index(self, documents, content_field='abstract', id_field='id',
                    min_df=5, max_df=0.95, use_idf=True):
        """Construit un index à partir des documents"""
        # Extraire les contenus et les IDs des documents
        contents = []
        self.document_ids = []

        for doc in documents:
            content = doc.get(content_field, '')
            if not content and 'title' in doc:  # Utiliser le titre si l'abstract est vide
                content = doc['title']

            if content:
                contents.append(content)
                self.document_ids.append(doc[id_field])

        # Créer un vectoriseur TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            use_idf=use_idf,
            sublinear_tf=True
        )

        # Créer un vectoriseur de comptage pour TF simple
        self.vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )

        # Construire la matrice TF-IDF
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)

        # Construire la matrice TF
        self.doc_term_matrix = self.vectorizer.fit_transform(contents)

        # Stocker les noms des termes
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

        return {
            'num_documents': len(self.document_ids),
            'vocabulary_size': len(self.feature_names)
        }

    def get_term_frequencies(self, top_n=20):
        """Retourne les termes les plus fréquents dans le corpus"""
        if self.doc_term_matrix is None:
            raise ValueError("L'index doit être construit avant d'obtenir les fréquences des termes")

        # Somme des occurrences de chaque terme dans tous les documents
        term_freqs = np.asarray(self.doc_term_matrix.sum(axis=0)).flatten()

        # Trier les termes par fréquence décroissante
        term_indices = term_freqs.argsort()[-top_n:][::-1]

        # Retourner les termes et leurs fréquences
        terms = [self.feature_names[i] for i in term_indices]
        freqs = [term_freqs[i] for i in term_indices]

        return dict(zip(terms, freqs))

    def get_document_vector(self, doc_id, use_tfidf=True):
        """Retourne le vecteur d'un document par son ID"""
        if self.document_ids is None:
            raise ValueError("L'index doit être construit avant de récupérer un vecteur document")

        try:
            idx = self.document_ids.index(doc_id)
            matrix = self.tfidf_matrix if use_tfidf else self.doc_term_matrix
            return matrix[idx].toarray().flatten()
        except ValueError:
            raise ValueError(f"Document avec ID {doc_id} non trouvé dans l'index")

    def get_index_stats(self):
        """Retourne des statistiques détaillées sur l'index"""
        if self.tfidf_matrix is None or self.doc_term_matrix is None:
            raise ValueError("L'index doit être construit avant de calculer les statistiques")

        # Calculer la densité des matrices
        tfidf_density = self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]) * 100
        tf_density = self.doc_term_matrix.nnz / (self.doc_term_matrix.shape[0] * self.doc_term_matrix.shape[1]) * 100

        stats = {
            'num_documents': len(self.document_ids),
            'vocabulary_size': len(self.feature_names),
            'tfidf_matrix_shape': self.tfidf_matrix.shape,
            'tf_matrix_shape': self.doc_term_matrix.shape,
            'tfidf_density': tfidf_density,
            'tf_density': tf_density
        }

        return stats