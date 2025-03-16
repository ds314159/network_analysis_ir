import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def plot_word_frequencies(word_freq, title="Top Words Frequency"):
    """
    Affiche un histogramme des mots les plus fréquents.

    Args:
        word_freq (list of tuple): Liste des mots et de leurs fréquences.
        title (str): Titre du graphique.
    """
    words, counts = zip(*word_freq)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.xticks(rotation=45)
    plt.xlabel("Mots")
    plt.ylabel("Fréquence")
    plt.title(title)
    plt.show()


def plot_network_graph(graph, title="Graph Visualization"):
    """
    Visualise un graphe avec NetworkX.

    Args:
        graph (networkx.Graph): Le graphe à afficher.
        title (str): Titre de la figure.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=300, node_color="skyblue", edge_color="gray")
    plt.title(title)
    plt.show()
