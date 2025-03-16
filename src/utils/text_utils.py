import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode

# Télécharger les stopwords si non présents
import nltk

nltk.download('stopwords')
nltk.download('punkt')


def clean_text(text):
    """
    Nettoie un texte en supprimant la ponctuation, les stopwords et en le normalisant.

    Args:
        text (str): Le texte brut.

    Returns:
        str: Texte nettoyé.
    """
    text = text.lower()  # Convertir en minuscules
    text = unidecode(text)  # Supprimer les accents
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    text = text.translate(str.maketrans('', '', string.punctuation))  # Supprimer la ponctuation
    tokens = word_tokenize(text)  # Tokenisation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Supprimer stopwords
    return ' '.join(tokens)


def get_top_n_words(corpus, n=10):
    """
    Retourne les mots les plus fréquents dans un corpus.

    Args:
        corpus (list of str): Liste des documents en texte brut.
        n (int): Nombre de mots les plus fréquents à extraire.

    Returns:
        list of tuple: Liste des n mots les plus fréquents avec leurs fréquences.
    """
    from collections import Counter
    all_words = ' '.join(corpus).split()
    word_freq = Counter(all_words)
    return word_freq.most_common(n)
