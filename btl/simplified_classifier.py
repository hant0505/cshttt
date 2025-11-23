# """
# BASS Simplified Active Learning Classifier
# - Chỉ dùng Logistic Regression từ scikit-learn
# - Không cần topic models phức tạp
# - Chọn document dựa trên entropy (uncertainty sampling)
# """

# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import adjusted_rand_score
# from sklearn.metrics.cluster import normalized_mutual_info_score
# import numpy as np
# import random

# def purity_score(y_true, y_pred):
#     """
#     Tính Purity score (tự định nghĩa thay vì dùng sklearn)
#     Purity = (số samples được classify đúng) / (tổng samples)
#     """
#     # Tạo confusion matrix
#     contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    
#     for i, true_label in enumerate(np.unique(y_true)):
#         for j, pred_label in enumerate(np.unique(y_pred)):
#             contingency_matrix[i][j] = np.sum((np.array(y_true) == true_label) & (np.array(y_pred) == pred_label))
    
#     # Purity = sum of max in each row / total
#     purity = np.sum(np.max(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
#     return purity


# class SimplifiedActiveLearning:
#     def __init__(self, X_unlabeled, feature_dim=10):
#         """
#         X_unlabeled: numpy array [num_docs, feature_dim]
#                      - có thể là random features hoặc embeddings đơn giản
#         """
#         self.X_unlabeled = X_unlabeled
#         self.num_docs = X_unlabeled.shape[0]
#         self.feature_dim = feature_dim
        
#         # Xử lý dữ liệu
#         self.scaler = StandardScaler()
#         self.X_scaled = self.scaler.fit_transform(X_unlabeled)
        
#         # Theo dõi documents đã label
#         self.labeled_doc_ids = set()
#         self.user_labels = {}  # {doc_id: label}
#         self.labeled_X = []    # Features của docs đã label
#         self.labeled_y = []    # Labels của docs đã label
        
#         # Classifier
#         self.classifier = None
#         self.classes = set()
#         self.entropy_scores = np.full(self.num_docs, float('inf'))
    
#     def recommend_document(self):
#         """
#         Active Learning: Chọn document có entropy cao nhất (uncertain)
#         """
#         if len(self.classes) < 2:
#             # Ngẫu nhiên nếu chưa có 2 labels
#             while True:
#                 doc_id = random.randint(0, self.num_docs - 1)
#                 if doc_id not in self.labeled_doc_ids:
#                     return doc_id, -1
#         else:
#             # Tính entropy cho tất cả documents
#             probas = self.classifier.predict_proba(self.X_scaled)
#             epsilon = 1e-9
#             entropy = -np.sum(probas * np.log(probas + epsilon), axis=1)
            
#             # Set entropy = -Inf cho documents đã label
#             for doc_id in self.labeled_doc_ids:
#                 entropy[doc_id] = float('-inf')
            
#             # Chọn document có entropy cao nhất
#             doc_id = np.argmax(entropy)
#             return doc_id, entropy[doc_id]
    
#     def label_document(self, doc_id, label):
#         """
#         User label một document
#         """
#         # Thay thế trong SimplifiedActiveLearning.label_document
#         self.user_labels[doc_id] = label
        
#         if doc_id not in self.labeled_doc_ids:
#             self.labeled_doc_ids.add(doc_id)
#             self.labeled_X.append(self.X_scaled[doc_id])
#             self.labeled_y.append(label)
#         else:
#         # update label cho doc đã label
#             idx = self.labeled_X.index(self.X_scaled[doc_id])
#             self.labeled_y[idx] = label

#         # Train classifier nếu có >=2 nhãn khác nhau
#         self.classes = set(self.labeled_y)
#         if len(self.classes) >= 2:
#             self.train_classifier()
#         # if doc_id not in self.labeled_doc_ids:
#         #     self.labeled_doc_ids.add(doc_id)
#         #     self.user_labels[doc_id] = label
#         #     self.labeled_X.append(self.X_scaled[doc_id])
#         #     self.labeled_y.append(label)
#         #     self.classes.add(label) # ← Thêm class mới
            
#         #     # Train classifier khi có >= 2 labels
#         #     if len(self.classes) >= 2:
#         #         self.train_classifier()
    
#     def train_classifier(self):
#         """
#         Train Logistic Regression classifier
#         """
#         X_train = np.array(self.labeled_X)
#         y_train = np.array(self.labeled_y)
        
#         self.classifier = LogisticRegression(
#             max_iter=200,
#             random_state=42,
#             multi_class='multinomial'
#         )
#         self.classifier.fit(X_train, y_train)
    
#     def get_predictions(self, doc_id):
#         """
#         Trả về top 3 predictions cho một document
#         """
#         if len(self.classes) < 2:
#             return ["Need at least 2 labels"], list(self.classes)
        
#         probas = self.classifier.predict_proba(self.X_scaled[doc_id:doc_id+1])
#         sorted_idx = np.argsort(probas[0])[::-1]
        
#         predictions = []
#         for idx in sorted_idx:
#             if probas[0][idx] > 0:
#                 label = self.classifier.classes_[idx]
#                 predictions.append((label, float(probas[0][idx])))
        
#         return predictions, list(self.classes)
    
#     def compute_metrics(self, ground_truth_labels):
#         """
#         Tính Purity, ARI, NMI
#         """
#         if len(self.classes) < 2:
#             return None, None, None
        
#         predictions = self.classifier.predict(self.X_scaled)
        
#         purity = purity_score(ground_truth_labels, predictions)
#         ari = adjusted_rand_score(ground_truth_labels, predictions)
#         nmi = normalized_mutual_info_score(ground_truth_labels, predictions)
        
#         return float(purity), float(ari), float(nmi)