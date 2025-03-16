import json
import os


def load_json(file_path):
    """
    Charge un fichier JSON en mémoire.

    Args:
        file_path (str): Chemin du fichier JSON.

    Returns:
        dict or list: Contenu du fichier JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json(data, file_path):
    """
    Sauvegarde un dictionnaire ou une liste dans un fichier JSON.

    Args:
        data (dict or list): Données à sauvegarder.
        file_path (str): Chemin du fichier de sortie.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def create_directory(directory):
    """
    Crée un répertoire s'il n'existe pas.

    Args:
        directory (str): Chemin du répertoire.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
