"""
Quick start: Load Mallet outputs + bills_preprocessed.json
"""
import json
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load doc-topics (Θ)
def load_doc_topics(doc_topics_file='model_LDA/MalletLda/modelFiles/doc-topics.txt'):
    """Load doc-topic distribution from Mallet."""
    doc_topics = []
    with open(doc_topics_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            topic_dist = np.array([float(x) for x in parts[2::2]])
            doc_topics.append(topic_dist)
    return np.array(doc_topics)

# 2. Load documents
def load_documents(json_path='bills_preprocessed.json'):
    """Load preprocessed documents."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Lấy processed_text từ mỗi document "CHECK file json là dict kphai list"
    if "processed_text" not in data:
        raise ValueError("processed_text key not found in JSON")

    # processed_text là dict: {id: "string cleaned text"}
    processed = data["processed_text"]

    # Lấy toàn bộ text dạng string
    texts = list(processed.values())

    # Đồng bộ hóa theo số doc mà Mallet tạo
    if doc_topics is not None:
        texts = texts[:len(doc_topics)]
    ## Do Total lines: 15000 Documents (excluding header) 
    #processed_text có 15000 văn bản.doc-topics.txt chỉ có 14999 dòng tài liệu. 1 tài liệu bị Mallet skip.
    return texts

# 3. Compute TF-IDF
def compute_tfidf(texts):
    """Compute TF-IDF matrix."""
    tfidf_vec = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
    tfidf_matrix = tfidf_vec.fit_transform(texts)
    return tfidf_matrix

# Main
doc_topics = load_doc_topics()
texts = load_documents()
tfidf_matrix = compute_tfidf(texts)

print(f"✓ Θ shape: {doc_topics.shape}")
print(f"✓ Texts: {len(texts)}")
print(f"✓ TF-IDF shape: {tfidf_matrix.shape}")

# Giờ dùng cho Enhanced Classifier (Step 3)
from classifier_enhanced import EnhancedActiveLearning

classifier = EnhancedActiveLearning(doc_topics, tfidf_matrix, combine_weight=0.5)
print("✓ Classifier ready!")