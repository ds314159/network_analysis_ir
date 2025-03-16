from sklearn.metrics import silhouette_score


def evaluate_topic_coherence(topic_model, documents):
    """
    Évalue la cohérence des topics en utilisant le score de silhouette.

    Le score de silhouette mesure à quel point les documents sont bien
    assignés à leurs topics. Un score élevé indique que les documents
    sont bien regroupés dans leurs topics respectifs et bien séparés
    des autres topics.

    Processus:
    1. Obtention des distributions de topics pour chaque document
    2. Détermination du topic dominant pour chaque document
    3. Calcul du score de silhouette basé sur les distances entre distributions

    Args:
        topic_model: Un modèle de topic entraîné
        documents: Les documents pour lesquels évaluer la cohérence

    Returns:
        Un score entre -1 et 1, où un score plus élevé indique des topics
        plus cohérents et mieux séparés
    """
    topic_distributions = topic_model.transform(documents)

    # Pour le score de silhouette, on utilise la distribution complète comme feature
    # et le topic le plus probable comme étiquette de cluster
    return silhouette_score(topic_distributions, topic_distributions.argmax(axis=1))