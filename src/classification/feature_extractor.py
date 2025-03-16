import networkx as nx
import numpy as np
import scipy.sparse
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif


class FeatureExtractor:
    def __init__(self, indexer=None, graph=None, max_features=2000, feature_selection='svd'):
        """
        Initialise l'extracteur de caractéristiques avec limitation de dimensions.

        Args:
            indexer: L'indexeur contenant les matrices TF-IDF et document-terme
            graph: Le graphe pour extraire des caractéristiques structurelles
            max_features: Nombre maximum de caractéristiques à conserver
            feature_selection: Méthode de sélection/réduction de caractéristiques
                               ('svd', 'chi2', 'mutual_info', 'none')
        """
        self.indexer = indexer
        self.graph = graph
        self.max_features = max_features
        self.feature_selection = feature_selection
        self.selector = None

        # Pour stocker la transformation utilisée
        # (utile pour transformer de nouvelles données de manière cohérente)
        self.transformation = None

        print(f"[INFO] FeatureExtractor initialisé avec max_features={max_features}, méthode={feature_selection}")

    def extract_text_features(self, doc_ids, use_tfidf=True, labels=None):
        """
        Extrait les caractéristiques textuelles des documents avec limitation de dimensions.

        Args:
            doc_ids: Liste des IDs de documents
            use_tfidf: Utiliser TF-IDF (True) ou fréquence brute (False)
            labels: Étiquettes de classe pour les méthodes de sélection supervisées (chi2, mutual_info)

        Returns:
            numpy.ndarray: Matrice de caractéristiques textuelles réduite
        """
        start_time = time.time()
        print(f"[INFO] Début de l'extraction des caractéristiques textuelles pour {len(doc_ids)} documents...")

        if self.indexer is None:
            raise ValueError("L'indexer doit être initialisé pour extraire des caractéristiques textuelles")

        # Indices des documents dans l'index
        print("[INFO] Récupération des indices de documents...")
        doc_indices = []
        missing_docs = 0
        for doc_id in doc_ids:
            if doc_id in self.indexer.document_ids:
                doc_indices.append(self.indexer.document_ids.index(doc_id))
            else:
                missing_docs += 1

        if missing_docs > 0:
            print(f"[ATTENTION] {missing_docs} documents non trouvés dans l'index!")

        print(f"[INFO] {len(doc_indices)} documents trouvés dans l'index")

        # Matrice de caractéristiques textuelles complète
        matrix = self.indexer.tfidf_matrix if use_tfidf else self.indexer.doc_term_matrix
        print(f"[INFO] Utilisation de la matrice {'TF-IDF' if use_tfidf else 'terme-document'} "
              f"de taille {matrix.shape}")

        # Extraire les caractéristiques des documents en question
        print("[INFO] Extraction des caractéristiques...")
        if scipy.sparse.issparse(matrix):
            features = matrix[doc_indices].toarray()
            print(f"[INFO] Conversion de la matrice sparse en array dense")
        else:
            features = matrix[doc_indices]

        print(f"[INFO] Caractéristiques extraites : {features.shape} (documents × termes)")

        # Vérifier si la réduction/sélection de caractéristiques est nécessaire
        if features.shape[1] <= self.max_features or self.feature_selection == 'none':
            elapsed_time = time.time() - start_time
            print(f"[INFO] Pas de réduction nécessaire. Dimension finale: {features.shape}")
            print(f"[INFO] Extraction terminée en {elapsed_time:.2f} secondes")
            return features

        # Appliquer la réduction/sélection de caractéristiques si nécessaire
        print(f"[INFO] Réduction des dimensions de {features.shape[1]} à {self.max_features} caractéristiques...")

        if self.transformation is None:
            if self.feature_selection == 'svd':
                # Réduction de dimension par SVD tronquée
                print("[INFO] Application de la SVD tronquée...")
                n_components = min(self.max_features, features.shape[1] - 1)
                self.transformation = TruncatedSVD(n_components=n_components)

                reduction_start = time.time()
                reduced_features = self.transformation.fit_transform(features)
                reduction_time = time.time() - reduction_start

                variance = self.transformation.explained_variance_ratio_.sum()
                print(f"[INFO] Réduction par SVD: {features.shape} -> {reduced_features.shape} "
                      f"(variance expliquée: {variance:.2f}, temps: {reduction_time:.2f}s)")

            elif self.feature_selection == 'chi2' and labels is not None:
                # Sélection des caractéristiques les plus discriminantes par chi2
                print("[INFO] Application de la sélection par chi2...")
                self.transformation = SelectKBest(chi2, k=self.max_features)

                reduction_start = time.time()
                reduced_features = self.transformation.fit_transform(features, labels)
                reduction_time = time.time() - reduction_start

                print(f"[INFO] Sélection par chi2: {features.shape} -> {reduced_features.shape} "
                      f"(temps: {reduction_time:.2f}s)")

            elif self.feature_selection == 'mutual_info' and labels is not None:
                # Sélection des caractéristiques par information mutuelle
                print("[INFO] Application de la sélection par information mutuelle...")
                self.transformation = SelectKBest(mutual_info_classif, k=self.max_features)

                reduction_start = time.time()
                reduced_features = self.transformation.fit_transform(features, labels)
                reduction_time = time.time() - reduction_start

                print(f"[INFO] Sélection par information mutuelle: {features.shape} -> {reduced_features.shape} "
                      f"(temps: {reduction_time:.2f}s)")

            else:
                # Par défaut, prendre simplement les premières caractéristiques
                print(f"[INFO] Utilisation des {self.max_features} premières caractéristiques")
                reduced_features = features[:, :self.max_features]

                # Créer une fonction de transformation simple pour la cohérence
                class SimpleSelector:
                    def transform(self, X):
                        return X[:, :self.max_features]

                self.transformation = SimpleSelector()
                self.transformation.max_features = self.max_features
        else:
            # Pour les nouvelles données, appliquer la transformation déjà apprise
            print("[INFO] Application de la transformation existante...")
            transformation_start = time.time()
            reduced_features = self.transformation.transform(features)
            transformation_time = time.time() - transformation_start
            print(f"[INFO] Transformation appliquée en {transformation_time:.2f}s")

        elapsed_time = time.time() - start_time
        print(f"[INFO] Extraction et réduction terminées en {elapsed_time:.2f} secondes")
        print(f"[INFO] Dimensions finales: {reduced_features.shape} (documents × caractéristiques)")

        return reduced_features

    def extract_text_embeddings_bert(self, doc_ids, model_name='bert-base-uncased', pooling_strategy='cls',
                                     max_length=512, batch_size=16):
        """
        Extrait les embeddings textuels des documents en utilisant un modèle BERT.

        Args:
            doc_ids: Liste des IDs de documents
            model_name: Nom du modèle BERT à utiliser (ex: 'bert-base-uncased', 'bert-large-uncased')
            pooling_strategy: Stratégie de pooling ('cls', 'mean', 'max')
            max_length: Longueur maximale des séquences en tokens
            batch_size: Taille des lots pour le traitement par lots

        Returns:
            numpy.ndarray: Matrice d'embeddings des documents
        """
        start_time = time.time()
        print(f"[INFO] Début de l'extraction des embeddings BERT pour {len(doc_ids)} documents...")

        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ImportError("Les bibliothèques 'transformers' et 'torch' sont nécessaires pour utiliser BERT")

        # Vérifier si l'indexeur est disponible
        if self.indexer is None:
            raise ValueError("L'indexer doit être initialisé pour récupérer le contenu des documents")

        # Charger le tokenizer et le modèle BERT
        print(f"[INFO] Chargement du modèle {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Utiliser GPU si disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Utilisation du périphérique: {device}")
        model = model.to(device)
        model.eval()  # Mode évaluation

        # Récupérer les textes des documents
        print("[INFO] Récupération du contenu des documents...")
        doc_texts = []
        missing_docs = 0
        found_docs = 0

        for doc_id in doc_ids:
            document = None
            if hasattr(self.indexer, 'get_document'):
                document = self.indexer.get_document(doc_id)

            if document:
                # Récupérer le texte (abstract ou contenu complet selon disponibilité)
                text = document.get('abstract', document.get('text', ''))
                doc_texts.append(text)
                found_docs += 1
            else:
                # Essayer de récupérer via une autre méthode si disponible
                doc_texts.append('')  # Ajouter un texte vide comme placeholder
                missing_docs += 1

        if missing_docs > 0:
            print(f"[ATTENTION] {missing_docs} documents non trouvés!")
        print(f"[INFO] {found_docs} documents récupérés avec succès")

        # Traitement par lots et extraction des embeddings
        print(f"[INFO] Extraction des embeddings avec stratégie de pooling '{pooling_strategy}'...")
        all_embeddings = []

        with torch.no_grad():  # Désactiver le calcul de gradient pour l'inférence
            for i in range(0, len(doc_texts), batch_size):
                batch_texts = doc_texts[i:i + batch_size]

                if i % 100 == 0 and i > 0:
                    print(f"[INFO] Traitement du lot {i // batch_size}/{len(doc_texts) // batch_size}...")

                # Tokenization avec padding et troncation
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)

                # Obtenir les embeddings de BERT
                outputs = model(**inputs)

                # Appliquer la stratégie de pooling sélectionnée
                if pooling_strategy == 'cls':
                    # Utiliser l'embedding du token [CLS]
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif pooling_strategy == 'mean':
                    # Calculer la moyenne des embeddings sur tous les tokens (en excluant les paddings)
                    attention_mask = inputs['attention_mask']
                    last_hidden = outputs.last_hidden_state

                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                    sum_mask = input_mask_expanded.sum(1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)  # Éviter la division par zéro

                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                elif pooling_strategy == 'max':
                    # Prendre le maximum sur la dimension des tokens
                    attention_mask = inputs['attention_mask']
                    last_hidden = outputs.last_hidden_state

                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    last_hidden[input_mask_expanded == 0] = -1e9  # Masquer les tokens de padding

                    batch_embeddings = torch.max(last_hidden, dim=1)[0].cpu().numpy()
                else:
                    raise ValueError(f"Stratégie de pooling '{pooling_strategy}' non supportée")

                all_embeddings.append(batch_embeddings)

        # Concaténer tous les embeddings
        embeddings = np.vstack(all_embeddings)
        print(f"[INFO] Embeddings extraits: {embeddings.shape} (documents × dimensions)")

        # Vérifier s'il faut réduire les dimensions
        if embeddings.shape[1] > self.max_features and self.feature_selection != 'none':
            print(f"[INFO] Réduction des dimensions de {embeddings.shape[1]} à {self.max_features}...")

            if self.transformation is None:
                if self.feature_selection == 'svd':
                    # Réduction par SVD
                    from sklearn.decomposition import TruncatedSVD
                    self.transformation = TruncatedSVD(n_components=self.max_features, random_state=42)
                    reduced_embeddings = self.transformation.fit_transform(embeddings)

                    variance = self.transformation.explained_variance_ratio_.sum()
                    print(f"[INFO] Réduction par SVD: {embeddings.shape} → {reduced_embeddings.shape} "
                          f"(variance expliquée: {variance:.2f})")
                else:
                    # Pour les autres méthodes, on utilise SVD par défaut pour les embeddings
                    print(
                        f"[INFO] Utilisation de SVD pour réduire les embeddings (méthode '{self.feature_selection}' ignorée)")
                    from sklearn.decomposition import TruncatedSVD
                    self.transformation = TruncatedSVD(n_components=self.max_features, random_state=42)
                    reduced_embeddings = self.transformation.fit_transform(embeddings)
            else:
                # Appliquer une transformation existante
                reduced_embeddings = self.transformation.transform(embeddings)
                print(f"[INFO] Transformation existante appliquée: {embeddings.shape} → {reduced_embeddings.shape}")

            embeddings = reduced_embeddings

        elapsed_time = time.time() - start_time
        print(f"[INFO] Extraction des embeddings terminée en {elapsed_time:.2f} secondes")

        return embeddings

    def extract_graph_features(self, node_ids):
        """Extrait les caractéristiques de graphe des nœuds"""
        start_time = time.time()
        print(f"[INFO] Début de l'extraction des caractéristiques de graphe pour {len(node_ids)} nœuds...")

        if self.graph is None:
            raise ValueError("Le graphe doit être initialisé pour extraire des caractéristiques structurelles")

        print(
            f"[INFO] Graphe disponible avec {self.graph.number_of_nodes()} nœuds et {self.graph.number_of_edges()} arêtes")

        features = []
        missing_nodes = 0

        for i, node_id in enumerate(node_ids):
            #if i % 100 == 0 and i > 0:
                #print(f"[INFO] Traitement du nœud {i}/{len(node_ids)}...")

            # Extraire des caractéristiques basiques du graphe
            node_features = []

            # Ajouter le degré du nœud
            if node_id in self.graph:
                node_features.append(self.graph.degree(node_id))

                # Coefficient de clustering
                try:
                    node_features.append(nx.clustering(self.graph, node_id))
                except Exception as e:
                    print(
                        f"[ATTENTION] Erreur lors du calcul du coefficient de clustering pour le nœud {node_id}: {str(e)}")
                    node_features.append(0)

                # Centralité
                try:
                    # Utiliser un sous-graphe pour les nœuds avec beaucoup de voisins
                    neighbors = list(self.graph.neighbors(node_id))
                    subgraph_nodes = neighbors + [node_id]
                    subgraph = self.graph.subgraph(subgraph_nodes)

                    # Centralité de betweenness locale
                    betweenness = nx.betweenness_centrality(subgraph)
                    node_features.append(betweenness.get(node_id, 0))

                    # Centralité de PageRank locale
                    pagerank = nx.pagerank(subgraph, alpha=0.85)
                    node_features.append(pagerank.get(node_id, 0))
                except Exception as e:
                    print(f"[ATTENTION] Erreur lors du calcul des centralités pour le nœud {node_id}: {str(e)}")
                    node_features.extend([0, 0])  # Ajouter des valeurs par défaut
            else:
                # Nœud non trouvé, ajouter des valeurs par défaut
                missing_nodes += 1
                node_features.extend([0, 0, 0, 0])

            features.append(node_features)

        if missing_nodes > 0:
            print(f"[ATTENTION] {missing_nodes} nœuds non trouvés dans le graphe!")

        features_array = np.array(features)

        elapsed_time = time.time() - start_time
        print(f"[INFO] Extraction des caractéristiques de graphe terminée en {elapsed_time:.2f} secondes")
        print(f"[INFO] Dimensions des caractéristiques de graphe: {features_array.shape}")

        return features_array

    def combine_features(self, text_features, graph_features=None, weights=(0.7, 0.3)):
        """Combine les caractéristiques textuelles et de graphe"""
        start_time = time.time()
        print("[INFO] Combinaison des caractéristiques...")

        # Normaliser les caractéristiques
        from sklearn.preprocessing import StandardScaler

        # Normaliser les caractéristiques textuelles
        print(f"[INFO] Normalisation des caractéristiques textuelles de forme {text_features.shape}...")
        scaler_text = StandardScaler()
        text_features_scaled = scaler_text.fit_transform(text_features)

        if graph_features is not None:
            # Normaliser les caractéristiques de graphe
            print(f"[INFO] Normalisation des caractéristiques de graphe de forme {graph_features.shape}...")
            scaler_graph = StandardScaler()
            graph_features_scaled = scaler_graph.fit_transform(graph_features)

            # Pondérer et combiner les caractéristiques
            text_weight, graph_weight = weights
            print(f"[INFO] Combinaison pondérée (texte: {text_weight}, graphe: {graph_weight})...")

            combined_features = np.hstack([
                text_features_scaled * text_weight,
                graph_features_scaled * graph_weight
            ])

            elapsed_time = time.time() - start_time
            print(f"[INFO] Combinaison terminée en {elapsed_time:.2f} secondes")
            print(f"[INFO] Dimensions finales: {combined_features.shape}")

            return combined_features
        else:
            print(
                "[INFO] Pas de caractéristiques de graphe fournies, utilisation uniquement des caractéristiques textuelles")

            elapsed_time = time.time() - start_time
            print(f"[INFO] Traitement terminé en {elapsed_time:.2f} secondes")

            return text_features_scaled