import yaml
import os
import json
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_config(config_path=None):
    """
    Charge la configuration depuis un fichier ou utilise des valeurs par défaut
    """
    # Configuration par défaut
    config = {
        "preprocessing": {
            "min_word_length": 3,
            "max_word_frequency": 0.95,
            "min_word_frequency": 5
        },
        "search_engine": {
            "similarity_measure": "cosine",
            "use_tfidf": True
        },
        "clustering": {
            "n_clusters": 8,
            "random_state": 42
        },
        "classification": {
            "test_size": 0.3,
            "random_state": 42
        }
    }

    # Si un chemin de fichier est fourni, charger la configuration depuis ce fichier
    if config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Mettre à jour la configuration par défaut avec les valeurs du fichier
                if file_config:
                    # Mettre à jour de manière récursive
                    def update_dict(d, u):
                        for k, v in u.items():
                            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                                update_dict(d[k], v)
                            else:
                                d[k] = v

                    update_dict(config, file_config)
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {e}")

    return config