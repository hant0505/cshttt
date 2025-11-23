# """
# BASS Simplified User Management
# - Quản lý session user
# - Load corpus + documents
# - Track labeling progress
# """

# import json
# import numpy as np
# from datetime import datetime
# import os
# from simplified_classifier import SimplifiedActiveLearning

# class SimplifiedUser:
#     def __init__(self, user_id, documents, document_embeddings=None):
#         """
#         user_id: string (e.g., "user_1")
#         documents: list of dicts [{id, text}, ...]
#         document_embeddings: numpy array [num_docs, embed_dim]
#                              nếu None → dùng random features
#         """
#         self.user_id = user_id
#         self.documents = {doc['id']: doc for doc in documents}
#         self.num_docs = len(documents)
        
#         # Tạo features nếu không có embeddings
#         if document_embeddings is None:
#             feature_dim = 10
#             document_embeddings = np.random.randn(self.num_docs, feature_dim)
        
#         self.embeddings = document_embeddings
        
#         # Active Learning classifier
#         self.classifier = SimplifiedActiveLearning(
#             document_embeddings,
#             feature_dim=document_embeddings.shape[1]
#         )
        
#         # Tracking
#         self.labeled_count = 0
#         self.start_time = datetime.now()
#         self.labels_history = []
#         self.metrics_history = []
    
#     def get_recommended_document(self):
#         """
#         Lấy document được recommended bởi Active Learning
#         """
#         doc_id, score = self.classifier.recommend_document()
#         return {
#             'doc_id': doc_id,
#             'text': self.documents[doc_id]['text'],
#             'uncertainty_score': float(score)
#         }
    
#     def label_document(self, doc_id, label):
#         """
#         User label một document
#         """
#         self.classifier.label_document(doc_id, label)
#         self.labeled_count += 1
        
#         self.labels_history.append({
#             'doc_id': doc_id,
#             'label': label,
#             'timestamp': datetime.now().isoformat()
#         })
        
#         return {
#             'status': 'labeled',
#             'labeled_count': self.labeled_count,
#             'total_docs': self.num_docs
#         }
    
#     def get_classifier_predictions(self, doc_id):
#         """
#         Lấy predictions từ classifier cho một document
#         """
#         predictions, all_labels = self.classifier.get_predictions(doc_id)
#         return {
#             'predictions': predictions,
#             'all_labels': all_labels
#         }
    
#     def compute_metrics(self, ground_truth_labels):
#         """
#         Tính metrics (nếu có ground truth)
#         """
#         purity, ari, nmi = self.classifier.compute_metrics(ground_truth_labels)
#         return {
#             'purity': purity,
#             'ari': ari,
#             'nmi': nmi
#         }
    
#     def get_session_summary(self):
#         """
#         Tóm tắt session user
#         """
#         duration = (datetime.now() - self.start_time).total_seconds()
#         return {
#             'user_id': self.user_id,
#             'labeled_count': self.labeled_count,
#             'total_docs': self.num_docs,
#             'duration_seconds': duration,
#             'labels_history': self.labels_history
#         }