import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Importing specific metrics to avoid reliance on star imports
from sklearn.metrics.cluster import adjusted_rand_score as ari_score 
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import random
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


def purity_score(y_true, y_pred):
    """Compute purity score for clustering evaluation."""
    contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    for i, true_label in enumerate(np.unique(y_true)):
        for j, pred_label in enumerate(np.unique(y_pred)):
            contingency_matrix[i][j] = np.sum((np.array(y_true) == true_label) & (np.array(y_pred) == pred_label))
    purity = np.sum(np.max(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
    return purity


class EnhancedActiveLearning:
    """
    Implements Step 3 of the BASS Workflow: Enhanced Active Learning Classifier.
    
    Combines:
    - Θ (doc-topic distribution from Mallet)
    - TF-IDF features (reduced via PCA)
    - User labels (incremental learning via model retraining)
    - Uncertainty sampling (entropy-based AL)
    [Image of Active Learning Workflow]
    """
    
    def __init__(self, doc_topics, tfidf_matrix, combine_weight=0.5, tfidf_vectorizer=None):
        """
        doc_topics: numpy array [num_docs, num_topics] (e.g., K=42)
        tfidf_matrix: sparse/dense matrix [num_docs, vocab_size] (e.g., V=1000)
        combine_weight: blend between topic features and TF-IDF (0-1).
        """
        """
        doc_topics: numpy [num_docs, num_topics]
        tfidf_matrix: sparse/dense TF-IDF [num_docs, vocab]
        combine_weight: trọng số giữa topics & tfidf
        tfidf_vectorizer: TfidfVectorizer đã fit, dùng cho semantic_search
        """
        self.doc_topics = doc_topics
        # Lưu bản gốc TF-IDF để semantic search
        self.tfidf_raw = tfidf_matrix
        self.tfidf_vectorizer = tfidf_vectorizer
        # Ensure TF-IDF is dense for PCA
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
        self.mlb = MultiLabelBinarizer()

        # Classifier
        self.classifier = None
    
    def _combine_features(self):
        """
        Combines Θ (doc-topics) and TF-IDF features using PCA reduction.
        
        The TF-IDF matrix is first reduced using PCA to match the dimensionality
        of the topic space (num_topics) before blending the two feature sets 
        linearly: w * topics + (1-w) * tfidf_reduced.
        """
        num_topics = self.doc_topics.shape[1]

        # 1. Reduce TF-IDF dimensionality (e.g., 14999 x 1000 -> 14999 x 42)
        print("INFO: Reducing TF-IDF dimensionality using PCA...")
        pca = PCA(n_components=num_topics, random_state=42)
        tfidf_reduced = pca.fit_transform(self.tfidf_matrix)

        # 2. Blend features
        combined = (
            self.combine_weight * self.doc_topics +
            (1 - self.combine_weight) * tfidf_reduced
        )

        print(f"✓ TF-IDF reduced shape: {tfidf_reduced.shape}")
        print(f"✓ Combined features shape: {combined.shape}")

        return combined

    def recommend_document(self):
        """
        Active Learning: select most uncertain document (entropy-based).
        If fewer than 2 classes are labeled, selects a document randomly.
        Returns: doc_id, uncertainty_score
        """
        if len(self.classes) < 2:
            # Random selection from unlabeled documents
            while True:
                doc_id = random.randint(0, self.num_docs - 1)
                if doc_id not in self.labeled_doc_ids:
                    return doc_id, -1
        else:            
            # Predict probabilities for all documents
            probas = self.classifier.predict_proba(self.features_scaled)
            
            # Calculate entropy: H(P) = -sum(p * log(p))
            epsilon = 1e-9
            entropy = -np.sum(probas * np.log(probas + epsilon), axis=1)
            
            # Disqualify already-labeled documents from being selected again
            for doc_id in self.labeled_doc_ids:
                entropy[doc_id] = float('-inf')
            
            # Select doc with highest entropy (most uncertain)
            doc_id = np.argmax(entropy)
            return doc_id, entropy[doc_id]
    
    def label_document(self, doc_id, label):
        """User labels a document, updating internal lists."""
            # Đảm bảo label luôn là list
        if not isinstance(label, list):
            label = [label]
        if doc_id not in self.labeled_doc_ids:
            self.labeled_doc_ids.add(doc_id)
            self.user_labels[doc_id] = label
            # Add scaled features and label
            self.labeled_x.append(self.features_scaled[doc_id])
            # Lưu nhãn dạng list (multi-label)
            self.labeled_y.append(label)
                    # Cập nhật tập các class
            for lb in label:
                self.classes.add(lb)            
            # Re-train classifier if >= 2 classes are present
            if len(self.classes) >= 2:
                self.train_classifier()
    
    def train_classifier(self):
        """
        Re-trains the SGDClassifier (acting as Logistic Regression) on the entire 
        accumulated labeled dataset. This is the "incremental learning" step, 
        ensuring the model is always optimally tuned to the current set of labels.
        """
        x_train = np.array(self.labeled_x)
        y_train = self.mlb.fit_transform(self.labeled_y)

        
        # Initialize/Rebuild SGDClassifier with log_loss for probability prediction
        # self.classifier = SGDClassifier(
        #     loss="log_loss", # Uses Logistic Regression loss
        #     max_iter=2000,
        #     tol=1e-3, 
        #     random_state=42
        # )
        sgd_clf = SGDClassifier(
            loss="log_loss", 
            penalty='l2', 
            tol=1e-3, 
            random_state=42, 
            learning_rate="optimal", 
            eta0=0.1, 
            validation_fraction=0.2, 
            alpha=0.000005  # Quan trọng: Regularization rất nhỏ
        )

        self.classifier = OneVsRestClassifier(sgd_clf)
        self.classifier.fit(x_train, y_train)

    def get_predictions(self, doc_id, top_k=3):
        """Return top-k labels + probability for multi-label classification."""
        if self.classifier is None:
            return [("Not enough labeled classes", 0.0)], [], 0.0

        try:
            probas = self.classifier.predict_proba(self.features_scaled[doc_id:doc_id+1])[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return [("Prediction failed", 0.0)], [], 0.0

        # Map index -> original label name
        labels = self.mlb.classes_

    # Combine
        label_probas = [(labels[i], float(probas[i])) for i in range(len(labels))]

    # Sort descending
        label_probas = sorted(label_probas, key=lambda x: x[1], reverse=True)

    # Top-k labels
        top_predictions = label_probas[:top_k]

    # Confidence = max probability
        confidence = max([p for _, p in label_probas]) if len(label_probas) else 0.0

        return top_predictions, list(labels), confidence

    def compute_metrics(self, ground_truth_labels):
        """Compute evaluation metrics against ground truth labels."""
        if len(self.classes) < 2 or self.classifier is None:
            print("DEBUG: Not enough classes or classifier not trained to compute metrics")
            return None, None, None
        # Lấy xác suất dự đoán cho TẤT CẢ tài liệu
        try:
            probas = self.classifier.predict_proba(self.features_scaled)
        except Exception as e:
            print(f"Metric error during predict_proba: {e}")
        return None, None, None
                # 1. Chuyển đổi Dự đoán Đa nhãn sang Đơn nhãn (Max Probability)
        # Lấy chỉ mục của nhãn có xác suất cao nhất
        max_prob_indices = np.argmax(probas, axis=1) 
        
        # Ánh xạ chỉ mục nhãn sang tên nhãn thực tế
        # self.mlb.classes_ chứa danh sách tên nhãn (string) theo thứ tự
        try:
            predicted_single_labels = [self.mlb.classes_[i] for i in max_prob_indices]
        except AttributeError:
             print("Metric error: MLB not initialized correctly or classes are empty.")
             return None, None, None
        
        # 2. Tính toán Metrics (sử dụng nhãn đơn)
        # NOTE: ground_truth_labels (nhãn ngẫu nhiên của bạn) cũng phải là nhãn đơn
        purity = purity_score(ground_truth_labels, predicted_single_labels)
        ari = ari_score(ground_truth_labels, predicted_single_labels)
        nmi = nmi_score(ground_truth_labels, predicted_single_labels)
        
        
        return float(purity), float(ari), float(nmi)
    
    def semantic_search(self, query, top_k=50):
        """Search theo TF-IDF + cosine similarity."""
        if self.tfidf_vectorizer is None or self.tfidf_raw is None:
            return []

        # 1. Query -> TF-IDF
        query_vec = self.tfidf_vectorizer.transform([query])

        # 2. Cosine similarity
        sims = cosine_similarity(query_vec, self.tfidf_raw)[0]

        # 3. Lấy doc có sim > 0
        candidate_ids = np.where(sims > 0)[0]

        ranked = sorted(
            [(int(i), float(sims[i])) for i in candidate_ids],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def get_session_info(self):
        """Get session summary."""
        return {
            'num_labeled': len(self.labeled_doc_ids),
            'num_classes': len(self.classes),
            'classes': list(self.classes),
            'labeled_docs': dict(self.user_labels)
        }