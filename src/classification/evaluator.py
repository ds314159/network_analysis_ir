import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Evaluator:
    def __init__(self):
        """
        Initialisation de l'évaluateur.
        """
        print("[INFO] Evaluator initialisé")

    def evaluate_classifier(self, true_labels, predicted_labels, class_names=None):
        """
        Évalue les performances d'un classificateur.

        Args:
            true_labels: Étiquettes réelles
            predicted_labels: Étiquettes prédites
            class_names: Noms des classes pour les visualisations

        Returns:
            dict: Métriques de performance
        """
        start_time = time.time()
        print(f"[INFO] Évaluation des performances du classificateur...")
        print(f"[INFO] Nombre d'échantillons: {len(true_labels)}")

        # Vérifier si les labels sont de la même longueur
        if len(true_labels) != len(predicted_labels):
            print(
                f"[ERREUR] Les étiquettes réelles ({len(true_labels)}) et prédites ({len(predicted_labels)}) n'ont pas la même longueur!")
            raise ValueError("Les étiquettes réelles et prédites doivent avoir la même longueur")

        # Calculer les métriques de performance
        print(f"[INFO] Calcul des métriques de classification...")

        # Utiliser les noms de classes fournis ou extraire les classes uniques
        if class_names is None:
            unique_classes = sorted(list(set(list(true_labels) + list(predicted_labels))))
            class_names = [str(c) for c in unique_classes]

        report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Calculer des métriques par classe
        class_metrics = {}
        for class_name in class_names:
            if class_name in report:
                class_metrics[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }

        # Afficher un résumé des résultats
        print(f"[INFO] Accuracy globale: {accuracy:.4f}")
        print(f"[INFO] F1-score macro: {report['macro avg']['f1-score']:.4f}")
        print(f"[INFO] F1-score pondéré: {report['weighted avg']['f1-score']:.4f}")

        # Visualiser la matrice de confusion
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Classe prédite')
        plt.ylabel('Classe réelle')
        plt.title('Matrice de confusion')
        plt.tight_layout()

        # Stocker les résultats
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'class_metrics': class_metrics,
            'confusion_matrix_plot': plt.gcf()
        }

        elapsed_time = time.time() - start_time
        print(f"[INFO] Évaluation terminée en {elapsed_time:.2f} secondes")

        return results

    def cross_validate(self, classifier, doc_ids, labels, n_splits=5, use_graph_features=True,
                       use_text_features=True, use_tfidf=True):
        """
        Effectue une validation croisée sur le classificateur.

        Args:
            classifier: Instance du classificateur
            doc_ids: Liste des IDs de documents
            labels: Étiquettes de classe pour les documents
            n_splits: Nombre de plis pour la validation croisée
            use_graph_features: Utiliser les caractéristiques de graphe
            use_text_features: Utiliser les caractéristiques textuelles
            use_tfidf: Utiliser TF-IDF (vs fréquence de termes)

        Returns:
            dict: Résultats de la validation croisée
        """
        start_time = time.time()
        print(f"[INFO] Début de la validation croisée ({n_splits} plis)...")
        print(f"[INFO] Configuration - texte: {use_text_features}, graphe: {use_graph_features}, tfidf: {use_tfidf}")

        # Extraire les caractéristiques
        feature_extraction_start = time.time()

        if use_text_features:
            print(f"[INFO] Extraction des caractéristiques textuelles pour {len(doc_ids)} documents...")
            text_features = classifier.feature_extractor.extract_text_features(doc_ids, use_tfidf)
        else:
            print(f"[INFO] Caractéristiques textuelles désactivées")
            text_features = np.zeros((len(doc_ids), 1))  # Au moins une colonne pour éviter les erreurs

        if use_graph_features:
            print(f"[INFO] Extraction des caractéristiques de graphe pour {len(doc_ids)} documents...")
            graph_features = classifier.feature_extractor.extract_graph_features(doc_ids)
        else:
            print(f"[INFO] Caractéristiques de graphe désactivées")
            graph_features = np.zeros((len(doc_ids), 1))  # Au moins une colonne pour éviter les erreurs

        feature_extraction_time = time.time() - feature_extraction_start
        print(f"[INFO] Extraction des caractéristiques terminée en {feature_extraction_time:.2f} secondes")

        # Combiner les caractéristiques
        print(f"[INFO] Combinaison des caractéristiques...")
        if use_text_features and use_graph_features:
            X = classifier.feature_extractor.combine_features(text_features, graph_features)
        elif use_text_features:
            X = text_features
        elif use_graph_features:
            X = graph_features
        else:
            raise ValueError("Au moins un type de caractéristique doit être activé")

        print(f"[INFO] Dimensions finales des caractéristiques: {X.shape}")

        # Convertir les étiquettes en entiers
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        unique_classes = label_encoder.classes_
        print(f"[INFO] Classes détectées: {len(unique_classes)} ({', '.join(map(str, unique_classes))})")

        # Définir le type de classificateur
        if classifier.model is None:
            print("[INFO] Modèle non initialisé, création d'un nouveau modèle pour la validation croisée")
            if hasattr(classifier, 'classifier_type'):
                model_type = classifier.classifier_type
            else:
                model_type = 'svm'  # Par défaut

            if model_type == 'svm':
                model = SVC(kernel='rbf', probability=True, random_state=42)
            elif model_type == 'rf':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == 'lr':
                model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
            else:
                raise ValueError(f"Type de classificateur non supporté: {model_type}")
        else:
            print("[INFO] Utilisation du modèle existant pour la validation croisée")
            model = classifier.model

        print(f"[INFO] Type de modèle utilisé: {model.__class__.__name__}")

        # Effectuer la validation croisée
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"[INFO] Exécution de la validation croisée stratifiée...")

        # Liste pour stocker les prédictions de chaque pli
        cv_predictions = []
        cv_true_labels = []

        # Dictionnaire pour stocker les scores par métrique
        cv_scores = {
            'accuracy': [],
            'f1_macro': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_weighted': []
        }

        # Effectuer manuellement la validation croisée pour collecter les prédictions
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            fold_start_time = time.time()
            print(f"[INFO] Pli {fold + 1}/{n_splits}...")

            # Diviser les données
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Entraîner le modèle
            train_start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - train_start

            # Prédire
            predict_start = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - predict_start

            # Collecter les prédictions
            cv_predictions.extend(y_pred)
            cv_true_labels.extend(y_test)

            # Calculer les métriques pour ce pli
            fold_accuracy = accuracy_score(y_test, y_pred)
            cv_scores['accuracy'].append(fold_accuracy)

            fold_report = classification_report(y_test, y_pred, output_dict=True)
            cv_scores['f1_macro'].append(fold_report['macro avg']['f1-score'])
            cv_scores['precision_macro'].append(fold_report['macro avg']['precision'])
            cv_scores['recall_macro'].append(fold_report['macro avg']['recall'])
            cv_scores['f1_weighted'].append(fold_report['weighted avg']['f1-score'])

            fold_time = time.time() - fold_start_time
            print(
                f"[INFO] Pli {fold + 1} - Précision: {fold_accuracy:.4f}, Temps d'entraînement: {train_time:.2f}s, Temps de prédiction: {predict_time:.2f}s, Temps total: {fold_time:.2f}s")

        # Calculer les moyennes et écarts-types
        results = {}
        for metric, scores in cv_scores.items():
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
            print(f"[INFO] {metric}: {results[f'{metric}_mean']:.4f} (±{results[f'{metric}_std']:.4f})")

        # Ajouter les résultats détaillés
        results['fold_accuracies'] = cv_scores['accuracy']

        # Convertir les indices en étiquettes d'origine pour l'évaluation globale
        cv_predictions_original = label_encoder.inverse_transform(cv_predictions)
        cv_true_labels_original = label_encoder.inverse_transform(cv_true_labels)

        # Évaluer globalement sur toutes les prédictions de validation croisée
        print(f"[INFO] Évaluation globale sur l'ensemble des prédictions de validation croisée...")
        global_evaluation = self.evaluate_classifier(cv_true_labels_original, cv_predictions_original)

        # Ajouter les résultats globaux
        results['global_accuracy'] = global_evaluation['accuracy']
        results['global_classification_report'] = global_evaluation['classification_report']
        results['global_confusion_matrix'] = global_evaluation['confusion_matrix']
        results['confusion_matrix_plot'] = global_evaluation.get('confusion_matrix_plot')

        elapsed_time = time.time() - start_time
        print(f"[INFO] Validation croisée terminée en {elapsed_time:.2f} secondes")

        return results

    def compare_clustering_with_classification(self, true_labels, cluster_labels):
        """
        Compare les résultats du clustering avec la classification supervisée.

        Args:
            true_labels: Étiquettes réelles (classification)
            cluster_labels: Étiquettes des clusters

        Returns:
            dict: Métriques de comparaison
        """
        start_time = time.time()
        print(f"[INFO] Comparaison des résultats de clustering avec la classification supervisée...")
        print(f"[INFO] Nombre d'échantillons: {len(true_labels)}")

        # Vérifier si les labels sont de la même longueur
        if len(true_labels) != len(cluster_labels):
            print(
                f"[ERREUR] Les étiquettes réelles ({len(true_labels)}) et de cluster ({len(cluster_labels)}) n'ont pas la même longueur!")
            raise ValueError("Les étiquettes réelles et de cluster doivent avoir la même longueur")

        # Afficher quelques statistiques
        unique_true = sorted(set(true_labels))
        unique_cluster = sorted(set(cluster_labels))
        print(f"[INFO] Classes de classification: {len(unique_true)} classes uniques")
        print(f"[INFO] Clusters détectés: {len(unique_cluster)} clusters uniques")

        # Calculer l'indice Rand ajusté (ARI)
        print(f"[INFO] Calcul de l'indice Rand ajusté (ARI)...")
        ari = adjusted_rand_score(true_labels, cluster_labels)

        # Calculer l'information mutuelle ajustée (AMI)
        print(f"[INFO] Calcul de l'information mutuelle ajustée (AMI)...")
        ami = adjusted_mutual_info_score(true_labels, cluster_labels)

        # Calculer l'homogénéité, la complétude et la V-mesure
        print(f"[INFO] Calcul des mesures d'homogénéité, complétude et V-mesure...")
        homogeneity = homogeneity_score(true_labels, cluster_labels)
        completeness = completeness_score(true_labels, cluster_labels)
        v_measure = v_measure_score(true_labels, cluster_labels)

        # Création d'une matrice de contingence
        print(f"[INFO] Création de la matrice de contingence...")
        contingency_df = pd.crosstab(
            pd.Series(true_labels, name='Classes réelles'),
            pd.Series(cluster_labels, name='Clusters'),
            normalize='index'  # Normaliser par ligne
        )

        # Visualiser la matrice de contingence
        plt.figure(figsize=(12, 8))
        sns.heatmap(contingency_df, annot=True, cmap='Blues', fmt='.2f')
        plt.title('Distribution des classes dans les clusters (normalisée par classe)')
        plt.tight_layout()

        results = {
            'ari': ari,
            'ami': ami,
            'homogeneity': homogeneity,
            'completeness': completeness,
            'v_measure': v_measure,
            'contingency_matrix': contingency_df.values,
            'contingency_plot': plt.gcf(),
            'true_classes': unique_true,
            'clusters': unique_cluster
        }

        # Afficher un résumé des résultats
        print(f"[INFO] Indice Rand ajusté (ARI): {ari:.4f} (entre -0.5 et 1, 1 = correspondance parfaite)")
        print(f"[INFO] Information mutuelle ajustée (AMI): {ami:.4f} (entre 0 et 1, 1 = correspondance parfaite)")
        print(f"[INFO] Homogénéité: {homogeneity:.4f} (entre 0 et 1, 1 = chaque cluster contient une seule classe)")
        print(
            f"[INFO] Complétude: {completeness:.4f} (entre 0 et 1, 1 = tous les points d'une classe sont dans un même cluster)")
        print(f"[INFO] V-mesure: {v_measure:.4f} (moyenne harmonique de l'homogénéité et de la complétude)")

        elapsed_time = time.time() - start_time
        print(f"[INFO] Comparaison terminée en {elapsed_time:.2f} secondes")

        return results

    def compare_feature_importance(self, classifier, doc_ids, labels, n_top_features=20,
                                   use_graph_features=True, use_text_features=True, use_tfidf=True):
        """
        Compare l'importance des différentes caractéristiques pour la classification.

        Args:
            classifier: Instance du classificateur
            doc_ids: Liste des IDs de documents
            labels: Étiquettes de classe pour les documents
            n_top_features: Nombre de caractéristiques principales à afficher
            use_graph_features: Utiliser les caractéristiques de graphe
            use_text_features: Utiliser les caractéristiques textuelles
            use_tfidf: Utiliser TF-IDF (vs fréquence de termes)

        Returns:
            dict: Résultats de l'analyse d'importance des caractéristiques
        """
        start_time = time.time()
        print(f"[INFO] Analyse de l'importance des caractéristiques...")

        # Extraction des caractéristiques
        if use_text_features:
            print(f"[INFO] Extraction des caractéristiques textuelles...")
            text_features = classifier.feature_extractor.extract_text_features(doc_ids, use_tfidf)

            # Récupérer les noms des caractéristiques textuelles
            if hasattr(classifier.feature_extractor.indexer, 'get_feature_names_out'):
                text_feature_names = classifier.feature_extractor.indexer.get_feature_names_out()
            elif hasattr(classifier.feature_extractor.indexer, 'get_feature_names'):
                text_feature_names = classifier.feature_extractor.indexer.get_feature_names()
            else:
                text_feature_names = [f"text_feature_{i}" for i in range(text_features.shape[1])]
        else:
            text_features = np.zeros((len(doc_ids), 0))
            text_feature_names = []

        if use_graph_features:
            print(f"[INFO] Extraction des caractéristiques de graphe...")
            graph_features = classifier.feature_extractor.extract_graph_features(doc_ids)

            # Noms des caractéristiques de graphe
            graph_feature_names = [
                "degré", "coefficient_clustering", "betweenness", "pagerank"
            ]

            # Ajuster si le nombre ne correspond pas
            if len(graph_feature_names) != graph_features.shape[1]:
                graph_feature_names = [f"graph_feature_{i}" for i in range(graph_features.shape[1])]
        else:
            graph_features = np.zeros((len(doc_ids), 0))
            graph_feature_names = []

        # Combinaison des caractéristiques
        if use_text_features and use_graph_features:
            X = classifier.feature_extractor.combine_features(text_features, graph_features)
            feature_names = list(text_feature_names) + list(graph_feature_names)
        elif use_text_features:
            X = text_features
            feature_names = list(text_feature_names)
        elif use_graph_features:
            X = graph_features
            feature_names = list(graph_feature_names)
        else:
            raise ValueError("Au moins un type de caractéristique doit être activé")

        print(f"[INFO] Dimensions finales des caractéristiques: {X.shape}")

        # Encodage des étiquettes
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        class_names = label_encoder.classes_

        # Créer un modèle Random Forest pour l'importance des caractéristiques
        print(f"[INFO] Entraînement d'un modèle Random Forest pour l'analyse d'importance...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)

        # Obtenir l'importance des caractéristiques
        importances = model.feature_importances_

        # Créer un DataFrame pour faciliter la visualisation
        feature_importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        })

        # Trier par importance
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

        # Afficher les caractéristiques les plus importantes
        print(f"[INFO] Top {min(n_top_features, len(feature_importance_df))} caractéristiques:")
        top_features = feature_importance_df.head(n_top_features)
        for i, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.6f}")

        # Visualiser l'importance des caractéristiques
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {n_top_features} caractéristiques les plus importantes')
        plt.tight_layout()

        results = {
            'feature_importance': feature_importance_df,
            'top_features': top_features,
            'importance_plot': plt.gcf()
        }

        elapsed_time = time.time() - start_time
        print(f"[INFO] Analyse d'importance terminée en {elapsed_time:.2f} secondes")

        return results

    def evaluate_error_analysis(self, true_labels, predicted_labels, doc_ids, text_field='abstract',
                                indexer=None, top_n_errors=5):
        """
        Effectue une analyse des erreurs de classification.

        Args:
            true_labels: Étiquettes réelles
            predicted_labels: Étiquettes prédites
            doc_ids: Liste des IDs de documents
            text_field: Nom du champ texte à afficher
            indexer: Indexeur contenant les documents
            top_n_errors: Nombre d'erreurs à analyser par type

        Returns:
            dict: Résultats de l'analyse des erreurs
        """
        start_time = time.time()
        print(f"[INFO] Analyse des erreurs de classification...")

        # Vérifier les données d'entrée
        if indexer is None:
            print(f"[ATTENTION] Indexeur non fourni, l'analyse détaillée des documents sera limitée")

        # Créer un DataFrame avec les étiquettes
        error_df = pd.DataFrame({
            'doc_id': doc_ids,
            'true_label': true_labels,
            'predicted_label': predicted_labels,
            'is_correct': (np.array(true_labels) == np.array(predicted_labels))
        })

        # Calculer le nombre d'erreurs par classe réelle
        errors_by_true_class = error_df[~error_df['is_correct']].groupby('true_label').size()
        total_by_true_class = error_df.groupby('true_label').size()
        error_rate_by_true_class = (errors_by_true_class / total_by_true_class * 100).fillna(0)

        # Calculer la matrice de confusion en pourcentage
        conf_matrix_pct = pd.crosstab(
            pd.Series(true_labels, name='Classe réelle'),
            pd.Series(predicted_labels, name='Classe prédite'),
            normalize='index'
        ) * 100

        # Afficher les taux d'erreur par classe
        print(f"[INFO] Taux d'erreur par classe réelle:")
        error_rate_df = pd.DataFrame({
            'classe': error_rate_by_true_class.index,
            'taux_erreur_%': error_rate_by_true_class.values,
            'erreurs': errors_by_true_class.values,
            'total': total_by_true_class.values
        })
        print(error_rate_df)

        # Analyser les cas les plus problématiques (confusions)
        confusion_analysis = []
        unique_true_classes = sorted(set(true_labels))

        for true_class in unique_true_classes:
            # Filtrer les erreurs pour cette classe réelle
            class_errors = error_df[(error_df['true_label'] == true_class) & (~error_df['is_correct'])]

            if len(class_errors) == 0:
                continue

            # Compter les prédictions erronées
            error_counts = class_errors['predicted_label'].value_counts()

            for pred_class, count in error_counts.items():
                confusion_rate = count / total_by_true_class[true_class] * 100
                confusion_analysis.append({
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'error_count': count,
                    'confusion_rate_%': confusion_rate
                })

        # Trier par nombre d'erreurs décroissant
        confusion_analysis = sorted(confusion_analysis, key=lambda x: x['error_count'], reverse=True)

        # Afficher les confusions les plus fréquentes
        print(f"[INFO] Confusions les plus fréquentes:")
        for i, confusion in enumerate(confusion_analysis[:10]):
            print(f"  {i + 1}. {confusion['true_class']} → {confusion['predicted_class']}: "
                  f"{confusion['error_count']} cas ({confusion['confusion_rate_%']:.1f}%)")

        # Exemples détaillés si un indexeur est fourni
        error_examples = []

        if indexer is not None and hasattr(indexer, 'get_document'):
            print(f"[INFO] Analyse des exemples d'erreurs:")

            # Trouver les confusions les plus fréquentes
            top_confusions = confusion_analysis[:min(5, len(confusion_analysis))]

            for confusion in top_confusions:
                true_class = confusion['true_class']
                pred_class = confusion['predicted_class']

                # Trouver des exemples pour cette confusion
                examples = error_df[
                    (error_df['true_label'] == true_class) &
                    (error_df['predicted_label'] == pred_class)
                    ].head(top_n_errors)

                # Obtenir les détails des documents
                for _, row in examples.iterrows():
                    doc_id = row['doc_id']
                    document = indexer.get_document(doc_id)

                    if document:
                        text = document.get(text_field, "")
                        if len(text) > 500:
                            text = text[:500] + "..."

                        error_examples.append({
                            'doc_id': doc_id,
                            'true_class': true_class,
                            'predicted_class': pred_class,
                            'text': text
                        })
                    else:
                        print(f"[ATTENTION] Document {doc_id} non trouvé dans l'indexeur")

            # Afficher quelques exemples
            print(f"[INFO] {len(error_examples)} exemples d'erreurs analysés")

        # Visualiser la matrice de confusion en pourcentage
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix_pct, annot=True, fmt='.1f', cmap='Blues')
        plt.title('Matrice de confusion (% par classe réelle)')
        plt.tight_layout()

        results = {
            'error_rate_by_class': error_rate_df,
            'confusion_analysis': confusion_analysis,
            'error_examples': error_examples,
            'confusion_matrix_pct': conf_matrix_pct,
            'confusion_pct_plot': plt.gcf()
        }

        elapsed_time = time.time() - start_time
        print(f"[INFO] Analyse des erreurs terminée en {elapsed_time:.2f} secondes")

        return results