import matplotlib.pyplot as plt
import seaborn as sns


def visualize_topics(topic_dict, model=None):
    """
    Affiche les topics sous forme de barres avec les poids réels des mots.

    Args:
        topic_dict: Dictionnaire {topic_name: [liste de mots]} à visualiser
        model: Le modèle qui a généré les topics (pour accéder aux poids)
    """
    for topic_id, words in topic_dict.items():
        # Obtenir l'index numérique du topic depuis son nom (ex: "Topic 0" -> 0)
        if isinstance(topic_id, str) and "Topic " in topic_id:
            topic_idx = int(topic_id.replace("Topic ", ""))
        else:
            topic_idx = topic_id

        # Créer une figure pour chaque topic
        plt.figure(figsize=(10, 6))

        # Pour LDA et NMF, on peut extraire les poids réels des mots
        if model and hasattr(model, 'model') and hasattr(model.model, 'components_'):
            # Obtenir les indices des mots dans le vocabulaire
            word_indices = [model.vectorizer.get_feature_names_out().tolist().index(word)
                            for word in words if word in model.vectorizer.get_feature_names_out()]

            # Obtenir les poids correspondants
            weights = [model.model.components_[topic_idx][idx] for idx in word_indices]

            # Normaliser les poids pour une meilleure visualisation
            if weights:
                weights = [w / max(weights) for w in weights]
        else:
            # Si on ne peut pas extraire les poids, utiliser des valeurs décroissantes
            weights = [1.0 - (i * 0.08) for i in range(len(words))]

        # Trier les mots par poids décroissant
        sorted_items = sorted(zip(words, weights), key=lambda x: x[1], reverse=True)
        sorted_words, sorted_weights = zip(*sorted_items) if sorted_items else ([], [])

        # Créer le graphique
        bars = plt.barh(range(len(sorted_words)), sorted_weights, align='center', color='steelblue')
        plt.yticks(range(len(sorted_words)), sorted_words)
        plt.xlabel('Poids relatif')
        plt.title(f"{topic_id}")
        plt.tight_layout()
        plt.show()