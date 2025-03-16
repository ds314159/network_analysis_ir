import re
import string
import unicodedata
import numpy as np
from typing import List, Dict, Any, Optional, Set
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

class DataCleaner:
    def __init__(self, config):
        self.config = config
        self.stop_words = self._load_stop_words()
        # Liste de chaînes suspectes à l'origine d'entrées fantômes dans les auteurs
        self.suspicious_author_names = ['br', 'hr', 'p', 'div', 'span']

    def clean_text(self, text):
        """Nettoie le texte (suppression des caractères spéciaux ert chiffres, mise en minuscule)"""
        # Vérifier si text est un float (NaN) et le convertir en chaîne vide
        if isinstance(text, float):
            return ""
        text = re.sub(r'\d', '', text)  # Supprime les chiffres
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text

    def remove_stop_words(self, tokens):
        """Supprime les mots vides d'une liste de tokens"""
        return [token for token in tokens if token not in self.stop_words]

    def normalize_text(self, text, method='lemmatization'):
        """Normalise le texte (stemming ou lemmatisation)"""
        if method == 'stemming':
            stemmer = PorterStemmer()
            return [stemmer.stem(token) for token in text.split()]
        elif method == 'lemmatization':
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(token) for token in text.split()]

    def _load_stop_words(self):
        """Charge la liste des mots vides"""
        return set(stopwords.words('english'))

    def clean_authors_list(self, authors_list):
        """Nettoie la liste des auteurs en retirant les entrées suspectes"""
        if not isinstance(authors_list, list):
            return []

        # Filtrer les auteurs suspects (balises HTML, chaînes trop courtes, etc.)
        cleaned_authors = []
        for author in authors_list:
            # Vérifier si l'auteur est une chaîne non vide
            if not isinstance(author, str) or not author.strip():
                continue

            # Vérifier si l'auteur n'est pas une balise HTML suspecte
            if author.lower() in self.suspicious_author_names:
                continue

            # Vérifier si l'auteur a une longueur raisonnable (au moins 3 caractères)
            if len(author.strip()) < 3:
                continue

            # Ajouter l'auteur à la liste nettoyée
            cleaned_authors.append(author.strip())

        return cleaned_authors