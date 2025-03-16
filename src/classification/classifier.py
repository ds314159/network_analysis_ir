import numpy as np
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from typing import Dict, List, Any, Optional, Tuple


class Classifier:
    def __init__(self, feature_extractor):
        """
        Initialise le classificateur.

        Args:
            feature_extractor: Instance de FeatureExtractor pour extraire les caractéristiques
        """
        self.feature_extractor = feature_extractor
        self.model = None
        self.classes = None
        self.label_encoder = None
        print(f"[INFO] Classifier initialisé avec le FeatureExtractor")

    def train_classifier(self, doc_ids, labels, classifier_type='svm',
                         use_graph_features=True, use_text_features=True,
                         use_tfidf=True, test_size=0.2, cross_validation=True):
        """
        Entraîne un classificateur sur les caractéristiques données.

        Args:
            doc_ids: Liste des IDs de documents
            labels: Étiquettes de classe pour les documents
            classifier_type: Type de classificateur ('svm', 'rf', 'lr')
            use_graph_features: Utiliser les caractéristiques de graphe
            use_text_features: Utiliser les caractéristiques textuelles
            use_tfidf: Utiliser TF-IDF (vs fréquence de termes)
            test_size: Proportion des données à utiliser pour le test
            cross_validation: Effectuer une validation croisée

        Returns:
            dict: Résultats de l'évaluation du modèle
        """
        start_time = time.time()
        print(f"[INFO] Début de l'entraînement du classificateur de type '{classifier_type}'...")
        print(f"[INFO] Configuration - texte: {use_text_features}, graphe: {use_graph_features}, tfidf: {use_tfidf}")

        # Encodage des étiquettes
        print(f"[INFO] Encodage des étiquettes...")
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        self.classes = self.label_encoder.classes_

        print(f"[INFO] Classes détectées: {len(self.classes)} ({', '.join(map(str, self.classes))})")

        # Extraction des caractéristiques
        if use_text_features:
            print(f"[INFO] Extraction des caractéristiques textuelles pour {len(doc_ids)} documents...")
            text_features = self.feature_extractor.extract_text_features(doc_ids, use_tfidf, labels=y)
        else:
            print(f"[INFO] Caractéristiques textuelles désactivées")
            text_features = np.zeros((len(doc_ids), 1))  # Au moins une colonne pour éviter les erreurs

        if use_graph_features:
            print(f"[INFO] Extraction des caractéristiques de graphe pour {len(doc_ids)} documents...")
            graph_features = self.feature_extractor.extract_graph_features(doc_ids)
        else:
            print(f"[INFO] Caractéristiques de graphe désactivées")
            graph_features = np.zeros((len(doc_ids), 1))  # Au moins une colonne pour éviter les erreurs

        # Combinaison des caractéristiques
        print(f"[INFO] Combinaison des caractéristiques...")
        if use_text_features and use_graph_features:
            X = self.feature_extractor.combine_features(text_features, graph_features)
        elif use_text_features:
            X = text_features
        elif use_graph_features:
            X = graph_features
        else:
            raise ValueError("Au moins un type de caractéristique doit être activé")

        print(f"[INFO] Dimensions finales des caractéristiques: {X.shape}")

        # Division train/test
        print(
            f"[INFO] Division des données en ensembles d'entraînement ({1 - test_size:.0%}) et de test ({test_size:.0%})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"[INFO] Ensembles d'entraînement: {X_train.shape[0]} exemples, test: {X_test.shape[0]} exemples")

        # Création du modèle
        print(f"[INFO] Initialisation du modèle {classifier_type}...")
        if classifier_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42, C=10, gamma='scale')
        elif classifier_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=None,
                                                min_samples_split=2, random_state=42, n_jobs=-1)
        elif classifier_type == 'lr':
            self.model = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                            random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Type de classificateur non supporté : {classifier_type}")

        # Validation croisée si demandée
        if cross_validation:
            print(f"[INFO] Exécution de la validation croisée (5-fold)...")
            cv_start_time = time.time()
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            cv_time = time.time() - cv_start_time

            print(f"[INFO] Scores de validation croisée: {cv_scores}")
            print(f"[INFO] Score moyen: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"[INFO] Validation croisée terminée en {cv_time:.2f} secondes")

        # Entraînement du modèle final
        print(f"[INFO] Entraînement du modèle final...")
        train_start_time = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - train_start_time

        print(f"[INFO] Modèle entraîné en {train_time:.2f} secondes")

        # Évaluation sur l'ensemble de test
        print(f"[INFO] Évaluation du modèle sur l'ensemble de test...")
        y_pred = self.model.predict(X_test)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=[str(c) for c in self.classes])
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"[INFO] Précision sur l'ensemble de test: {accuracy:.4f}")
        print(f"[INFO] Rapport de classification:\n{class_report}")

        # Résumé des résultats
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'model_type': classifier_type,
            'features_used': {
                'text': use_text_features,
                'graph': use_graph_features,
                'tfidf': use_tfidf
            },
            'feature_dimensions': X.shape[1],
            'training_time': train_time
        }

        if cross_validation:
            results['cv_scores'] = cv_scores
            results['cv_mean'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()

        total_time = time.time() - start_time
        print(f"[INFO] Processus d'entraînement terminé en {total_time:.2f} secondes")

        return results

    def predict(self, doc_ids, use_graph_features=True, use_text_features=True, use_tfidf=True):
        """
        Prédit les classes pour les documents donnés.

        Args:
            doc_ids: Liste des IDs de documents
            use_graph_features: Utiliser les caractéristiques de graphe
            use_text_features: Utiliser les caractéristiques textuelles
            use_tfidf: Utiliser TF-IDF (vs fréquence de termes)

        Returns:
            list: Liste des classes prédites
        """
        start_time = time.time()
        print(f"[INFO] Prédiction pour {len(doc_ids)} documents...")

        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")

        # Extraction des caractéristiques
        if use_text_features:
            print(f"[INFO] Extraction des caractéristiques textuelles...")
            text_features = self.feature_extractor.extract_text_features(doc_ids, use_tfidf)
        else:
            text_features = np.zeros((len(doc_ids), 1))

        if use_graph_features:
            print(f"[INFO] Extraction des caractéristiques de graphe...")
            graph_features = self.feature_extractor.extract_graph_features(doc_ids)
        else:
            graph_features = np.zeros((len(doc_ids), 1))

        # Combinaison des caractéristiques
        print(f"[INFO] Combinaison des caractéristiques...")
        if use_text_features and use_graph_features:
            X = self.feature_extractor.combine_features(text_features, graph_features)
        elif use_text_features:
            X = text_features
        elif use_graph_features:
            X = graph_features
        else:
            raise ValueError("Au moins un type de caractéristique doit être activé")

        # Prédiction
        print(f"[INFO] Calcul des prédictions...")
        y_pred = self.model.predict(X)

        # Conversion en classes d'origine
        predictions = [self.classes[i] for i in y_pred]

        elapsed_time = time.time() - start_time
        print(f"[INFO] Prédictions terminées en {elapsed_time:.2f} secondes")

        return predictions

    def predict_proba(self, doc_ids, use_graph_features=True, use_text_features=True, use_tfidf=True):
        """
        Prédit les probabilités de classe pour les documents donnés.

        Args:
            doc_ids: Liste des IDs de documents
            use_graph_features: Utiliser les caractéristiques de graphe
            use_text_features: Utiliser les caractéristiques textuelles
            use_tfidf: Utiliser TF-IDF (vs fréquence de termes)

        Returns:
            list: Liste de dictionnaires {classe: probabilité}
        """
        start_time = time.time()
        print(f"[INFO] Calcul des probabilités pour {len(doc_ids)} documents...")

        if self.model is None or not hasattr(self.model, 'predict_proba'):
            raise ValueError("Le modèle doit être entraîné et supporter predict_proba")

        # Extraction des caractéristiques
        if use_text_features:
            print(f"[INFO] Extraction des caractéristiques textuelles...")
            text_features = self.feature_extractor.extract_text_features(doc_ids, use_tfidf)
        else:
            text_features = np.zeros((len(doc_ids), 1))

        if use_graph_features:
            print(f"[INFO] Extraction des caractéristiques de graphe...")
            graph_features = self.feature_extractor.extract_graph_features(doc_ids)
        else:
            graph_features = np.zeros((len(doc_ids), 1))

        # Combinaison des caractéristiques
        print(f"[INFO] Combinaison des caractéristiques...")
        if use_text_features and use_graph_features:
            X = self.feature_extractor.combine_features(text_features, graph_features)
        elif use_text_features:
            X = text_features
        elif use_graph_features:
            X = graph_features
        else:
            raise ValueError("Au moins un type de caractéristique doit être activé")

        # Prédiction des probabilités
        print(f"[INFO] Calcul des probabilités...")
        probas = self.model.predict_proba(X)

        # Conversion en dictionnaires classe -> probabilité
        predictions = []
        for proba in probas:
            pred_dict = {self.classes[i]: float(p) for i, p in enumerate(proba)}
            predictions.append(pred_dict)

        elapsed_time = time.time() - start_time
        print(f"[INFO] Calcul des probabilités terminé en {elapsed_time:.2f} secondes")

        return predictions

    def save_model(self, filepath):
        """
        Sauvegarde le modèle entraîné sur le disque.

        Args:
            filepath: Chemin du fichier où sauvegarder le modèle
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant d'être sauvegardé")

        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        print(f"[INFO] Sauvegarde du modèle vers {filepath}...")

        # Sauvegarder le modèle et ses informations associées
        model_info = {
            'model': self.model,
            'classes': self.classes,
            'label_encoder': self.label_encoder
        }

        joblib.dump(model_info, filepath)
        print(f"[INFO] Modèle sauvegardé avec succès")

    @classmethod
    def load_model(cls, filepath, feature_extractor):
        """
        Charge un modèle sauvegardé.

        Args:
            filepath: Chemin du fichier modèle
            feature_extractor: Instance de FeatureExtractor

        Returns:
            Classifier: Instance avec le modèle chargé
        """
        print(f"[INFO] Chargement du modèle depuis {filepath}...")

        # Créer une nouvelle instance
        classifier = cls(feature_extractor)

        # Charger le modèle
        model_info = joblib.load(filepath)
        classifier.model = model_info['model']
        classifier.classes = model_info['classes']
        classifier.label_encoder = model_info['label_encoder']

        print(f"[INFO] Modèle chargé avec succès")
        print(f"[INFO] Classes chargées: {len(classifier.classes)} ({', '.join(map(str, classifier.classes))})")

        return classifier

    def evaluate_model_variants(self, doc_ids, labels, test_size=0.2):
        """
        Évalue différentes variantes du modèle pour comparer les performances.

        Args:
            doc_ids: Liste des IDs de documents
            labels: Étiquettes de classe pour les documents
            test_size: Proportion des données à utiliser pour le test

        Returns:
            dict: Résultats comparatifs des différentes variantes
        """
        print(f"[INFO] Évaluation comparative des variantes du modèle...")

        # Définir les configurations à tester
        configs = [
            {'name': 'Texte uniquement (TF-IDF)', 'text': True, 'graph': False, 'tfidf': True},
            {'name': 'Texte uniquement (TF)', 'text': True, 'graph': False, 'tfidf': False},
            {'name': 'Graphe uniquement', 'text': False, 'graph': True, 'tfidf': False},
            {'name': 'Texte (TF-IDF) + Graphe', 'text': True, 'graph': True, 'tfidf': True},
            {'name': 'Texte (TF) + Graphe', 'text': True, 'graph': True, 'tfidf': False}
        ]

        # Modèles à tester
        models = ['svm', 'rf', 'lr']

        results = {}

        for model in models:
            model_results = {}
            print(f"\n[INFO] Évaluation des configurations pour le modèle {model.upper()}...")

            for config in configs:
                print(f"\n[INFO] Configuration: {config['name']}...")

                # Entraîner et évaluer le modèle
                result = self.train_classifier(
                    doc_ids=doc_ids,
                    labels=labels,
                    classifier_type=model,
                    use_text_features=config['text'],
                    use_graph_features=config['graph'],
                    use_tfidf=config['tfidf'],
                    test_size=test_size,
                    cross_validation=True
                )

                # Stocker les résultats
                model_results[config['name']] = {
                    'accuracy': result['accuracy'],
                    'cv_mean': result.get('cv_mean', None),
                    'training_time': result['training_time']
                }

            results[model] = model_results

        # Afficher un résumé comparatif
        print("\n[INFO] Résumé des performances:")
        for model, model_results in results.items():
            print(f"\n  Modèle: {model.upper()}")

            for config_name, metrics in model_results.items():
                print(f"    {config_name}:")
                print(f"      Précision: {metrics['accuracy']:.4f}")
                if metrics['cv_mean'] is not None:
                    print(f"      CV moyenne: {metrics['cv_mean']:.4f}")
                print(f"      Temps d'entraînement: {metrics['training_time']:.2f}s")

        return results