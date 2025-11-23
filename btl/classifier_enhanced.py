"""
BASS Workflow - Step 3: Enhanced Active Learning Classifier
- Combine Θ (doc-topic) + TF-IDF features
- Uncertainty sampling (entropy-based)
- Incremental learning as user labels documents
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import random
from sklearn.decomposition import PCA


def purity_score(y_true, y_pred):
    """Compute purity score."""
    contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    for i, true_label in enumerate(np.unique(y_true)):
        for j, pred_label in enumerate(np.unique(y_pred)):
            contingency_matrix[i][j] = np.sum((np.array(y_true) == true_label) & (np.array(y_pred) == pred_label))
    purity = np.sum(np.max(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
    return purity


class EnhancedActiveLearning:
    """
    Combines:
    - Θ (doc-topic distribution from Mallet)
    - TF-IDF features
    - User labels (incremental learning)
    - Uncertainty sampling (entropy-based AL)
    """
    
    def __init__(self, doc_topics, tfidf_matrix, combine_weight=0.5):
        """
        doc_topics: numpy array [num_docs, num_topics]
        tfidf_matrix: sparse matrix [num_docs, vocab_size]
        combine_weight: blend between topic features and TF-IDF (0-1)
        """
        self.doc_topics = doc_topics
        self.tfidf_matrix = tfidf_matrix.toarray() if hasattr(tfidf_matrix, 'toarray') else tfidf_matrix
        self.combine_weight = combine_weight
        self.num_docs = doc_topics.shape[0]
        
        # Normalize and combine features
        self.features = self._combine_features()
        
        # Scaler for classifier
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        # Tracking labeled documents
        self.labeled_doc_ids = set()
        self.user_labels = {}
        self.labeled_x = []
        self.labeled_y = []
        self.classes = set()
        
        # Classifier
        self.classifier = None
    
    # def _combine_features(self):
    #     """Combine Θ + TF-IDF features."""
    #     # Normalize TF-IDF
    #     tfidf_norm = self.tfidf_matrix / (np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True) + 1e-9)
        
    #     # Combine: w * topic_features + (1-w) * tfidf_features
    #     combined = (
    #         self.combine_weight * self.doc_topics + 
    #         (1 - self.combine_weight) * tfidf_norm
    #     )
    #     return combined

    def _combine_features(self):
        """
    Combine Θ + TF-IDF using PCA:
    - Reduce TF-IDF (14999 × 1000) → 14999 × 42
    - Blend two feature spaces together
        """
        num_topics = self.doc_topics.shape[1]

    # Giảm chiều TF-IDF từ 1000 → 42
        pca = PCA(n_components=num_topics, random_state=42)
        tfidf_reduced = pca.fit_transform(self.tfidf_matrix)

    # Kết hợp: w * topics + (1-w) * tfidf_reduced
        combined = (
            self.combine_weight * self.doc_topics +
            (1 - self.combine_weight) * tfidf_reduced
        )

        print("✓ TF-IDF reduced shape:", tfidf_reduced.shape)
        print("✓ Combined features shape:", combined.shape)

        return combined

    def recommend_document(self):
        """
        Active Learning: select most uncertain document (entropy-based).
        Returns: doc_id, uncertainty_score
        """
        if len(self.classes) < 2:
            # Not enough labels yet, random selection
            while True:
                doc_id = random.randint(0, self.num_docs - 1)
                if doc_id not in self.labeled_doc_ids:
                    return doc_id, -1
        else:
            """
            ĐÂY là entropy uncertainty sampling, và bạn còn xử lý:
                -Bỏ các doc đã được label
                -Khi chưa đủ lớp thì random → hợp lý
            """
            # Compute entropy for all documents
            probas = self.classifier.predict_proba(self.features_scaled)
            epsilon = 1e-9
            entropy = -np.sum(probas * np.log(probas + epsilon), axis=1)
            
            # Set entropy = -Inf for already-labeled docs
            for doc_id in self.labeled_doc_ids:
                entropy[doc_id] = float('-inf')
            
            # Select doc with highest entropy
            doc_id = np.argmax(entropy)
            return doc_id, entropy[doc_id]
    
    def label_document(self, doc_id, label):
        """User labels a document."""
        if doc_id not in self.labeled_doc_ids:
            self.labeled_doc_ids.add(doc_id)
            self.user_labels[doc_id] = label
            self.labeled_x.append(self.features_scaled[doc_id])
            self.labeled_y.append(label)
            self.classes.add(label)
            
            # Train classifier if >= 2 classes
            if len(self.classes) >= 2:
                self.train_classifier()
    
    def train_classifier(self):
        """Train Logistic Regression on labeled data."""
        x_train = np.array(self.labeled_x)
        y_train = np.array(self.labeled_y)
        
        self.classifier = LogisticRegression(
            max_iter=200,
            random_state=42,
            multi_class='multinomial'
        )
        self.classifier.fit(x_train, y_train)
    
    def get_predictions(self, doc_id, top_k=3):
        """Get top-k predictions for a document."""
        if len(self.classes) < 2:
            return ["Need at least 2 labels"], list(self.classes)
        
        probas = self.classifier.predict_proba(self.features_scaled[doc_id:doc_id+1])
        sorted_idx = np.argsort(probas[0])[::-1]
        
        predictions = []
        for idx in sorted_idx:
            if probas[0][idx] > 0:
                label = self.classifier.classes_[idx]
                predictions.append((label, float(probas[0][idx])))
        
        return predictions[:top_k], list(self.classes)
    
    def compute_metrics(self, ground_truth_labels):
        """Compute evaluation metrics."""
        if len(self.classes) < 2:
            print("DEBUG: Not enough classes to compute metrics")
            return None, None, None

        if self.classifier is None:
            print("Classifier not trained yet")
            return None, None, None
        
        predictions = self.classifier.predict(self.features_scaled)
        # print("Predictions:", predictions[:10])
        purity = purity_score(ground_truth_labels, predictions)
        ari = adjusted_rand_score(ground_truth_labels, predictions)
        nmi = normalized_mutual_info_score(ground_truth_labels, predictions)
        
        return float(purity), float(ari), float(nmi)
    
    def get_session_info(self):
        """Get session summary."""
        return {
            'num_labeled': len(self.labeled_doc_ids),
            'num_classes': len(self.classes),
            'classes': list(self.classes),
            'labeled_docs': dict(self.user_labels)
        }