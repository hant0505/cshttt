from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from pathlib import Path
import uuid
import copy
from datetime import datetime

# Imports
from init_classifier import load_doc_topics, load_documents, compute_tfidf
from classifier_enhanced import EnhancedActiveLearning
from llm_suggestions import LLMSuggestionsOpenRouter

app = Flask(__name__)
CORS(app)

# Load base data
print("Loading data...")
doc_topics = load_doc_topics()
# texts = load_documents()
texts_processed, texts_raw = load_documents()
tfidf_matrix, tfidf_vectorizer = compute_tfidf(texts_processed)
BASE_CLASSIFIER = EnhancedActiveLearning(doc_topics, tfidf_matrix, combine_weight=0.5,tfidf_vectorizer=tfidf_vectorizer)
print(f"✓ Base classifier loaded")

# User sessions
USERS = {}

# LLM
LLM_API_KEY = "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"
llm_suggester = LLMSuggestionsOpenRouter(LLM_API_KEY, model="meta-llama/llama-3.1-8b-instruct")

# ============= ENDPOINTS =============

@app.route('/create_user', methods=['POST', 'GET'])
def create_user():
    """Tạo session user mới"""
    try:
        user_id = str(uuid.uuid4())[:8]
        USERS[user_id] = {
            'classifier': copy.deepcopy(BASE_CLASSIFIER),
            'documents_processed': texts_processed,
            'documents_raw': texts_raw,
            'labeled_count': 0,
            'labeled_history': [],  # Track labeled docs
            'skipped': set(),       # Track skipped docs
            'current_doc_id': None
        }
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'total_documents': len(texts_processed)
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_recommended_document', methods=['GET'])
def get_recommended_document():
    """Lấy doc tiếp theo (Active Learning)"""
    try:
        user_id = request.args.get('user_id')
        if user_id not in USERS:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
        classifier = USERS[user_id]['classifier']
        # documents = USERS[user_id]['documents']
        #-> hiện cho ng dùng raw
        documents = USERS[user_id]['documents_raw']
        doc_id, entropy = classifier.recommend_document()
        
        # FIX: Convert numpy int64 to python int
        doc_id = int(doc_id)

        # Store current doc
        USERS[user_id]['current_doc_id'] = doc_id
        
        # doc_text = documents[doc_id] if isinstance(documents[doc_id], str) else documents[doc_id].get('text', '')
        doc_text = documents[doc_id] #raw
        return jsonify({
            'status': 'success',
            'doc_id': doc_id,
            'text': doc_text,
            'entropy_score': float(entropy) if entropy != -1 else None
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_document_information', methods=['GET'])
def get_document_information():
    """Tóm tắt + LLM gợi ý labels cho 1 document"""
    try:
        user_id = request.args.get('user_id')
        doc_id = int(request.args.get('doc_id'))
        
        if user_id not in USERS:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
        doc_text = USERS[user_id]['documents_raw'][doc_id]
        if isinstance(doc_text, dict):
            doc_text = doc_text.get('text', '')
        
        # Get LLM suggestions
        suggestions = llm_suggester.suggest_labels_for_doc(doc_text, top_k=3)
        llm_labels = [label for label, _ in suggestions]
        
        # Get summary (optional)
        summary = llm_suggester.generate_label_summary("Document", [doc_text])
        
        return jsonify({
            'status': 'success',
            'doc_id': doc_id,
            'text': doc_text[:300],  # Return truncated text preview if needed
            'summary': summary,
            'llm_labels': llm_labels,
            'suggestions': [{'label': l, 'confidence': c} for l, c in suggestions]
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/label_document', methods=['POST'])
def label_document():
    """Lưu label cho document"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        doc_id = int(data.get('doc_id'))
        label = data.get('label')
        
        if user_id not in USERS:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
        classifier = USERS[user_id]['classifier']
        classifier.label_document(doc_id, label)

        # update counter
        USERS[user_id]['labeled_count'] += 1

        # update history
        USERS[user_id]['labeled_history'].append({
            'doc_id': doc_id,
            'label': label,
            'timestamp': datetime.now().isoformat()
        })

        # update current doc here
        USERS[user_id]['current_doc_id'] = doc_id
        
        return jsonify({
            'status': 'success',
            'message': f'Labeled doc {doc_id} as {label}',
            'labeled_count': USERS[user_id]['labeled_count'],
            'current_doc_id': doc_id
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/back_document', methods=['POST'])
def back_document():
    data = request.get_json()
    user_id = data.get('user_id')

    if user_id not in USERS:
        return jsonify({"status": "error", "message": "User not found"}), 404

    history = USERS[user_id]['labeled_history']
    if not history:
        return jsonify({"status": "error", "message": "No previous document"}), 400

    last = history.pop()
    doc_id = last['doc_id']
    label = last['label']

    # Undo: rebuild classifier from scratch
    classifier = copy.deepcopy(BASE_CLASSIFIER)
    for h in history:
        classifier.label_document(h["doc_id"], h["label"])

    USERS[user_id]['classifier'] = classifier
    USERS[user_id]['labeled_count'] -= 1
    USERS[user_id]['current_doc_id'] = doc_id

    # Return doc content
    doc_text = USERS[user_id]['documents_raw'][doc_id]
    if isinstance(doc_text, dict):
        doc_text = doc_text.get("text", "")

    return jsonify({
        "status": "success",
        "doc_id": int(doc_id), # Ensure int
        "text": doc_text
    })

@app.route('/search', methods=['POST'])
def search_documents():
    data = request.get_json()
    user_id = data.get('user_id')
    query = data.get('query')

    classifier = USERS[user_id]['classifier']
    
    # Check if method exists
    if hasattr(classifier, 'semantic_search'):
        scores = classifier.semantic_search(query, top_k=50)
        # FIX: Convert doc_id to int
        results = [{'id': int(doc_id), 'score': float(score)} for doc_id, score in scores]
    else:
        results = []

    return jsonify({
        'status': 'success',
        'results': results
    })

# @app.route('/documents', methods=['GET'])
# def get_documents():
#     user_id = request.args.get('user_id')

#     if user_id is None or user_id not in USERS:
#         return jsonify({
#             "status": "error",
#             "message": "Invalid or missing user_id"
#         }), 400

#     documents = USERS[user_id]['documents_raw']

#     return jsonify({
#         "status": "success",
#         "documents": [
#             {"id": i, "text": documents[i] if isinstance(documents[i], str) else documents[i].get('text', '')}
#             for i in range(len(documents))
#         ]
#     })
@app.route('/documents', methods=['GET'])
def get_documents():
    user_id = request.args.get('user_id')

    if user_id is None or user_id not in USERS:
        return jsonify({
            "status": "error",
            "message": "Invalid or missing user_id"
        }), 400

    documents = USERS[user_id]['documents_raw']   # ✅ DÙNG RAW

    return jsonify({
        "status": "success",
        "documents": [
            {
                "id": i,
                "text": documents[i]
            }
            for i in range(len(documents))
        ]
    })

@app.route('/get_topic_list', methods=['GET', 'OPTIONS'])
def get_topic_list():
    """Danh sách các topic đã được user tạo + docs thuộc topic"""
    try:
        user_id = request.args.get('user_id')

        if user_id not in USERS:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
        classifier = USERS[user_id]['classifier']
        num_docs = len(USERS[user_id]['documents_processed'])

        # 1. Danh sách các chủ đề đã được người dùng tạo (nhãn thủ công)
        topics = list(classifier.classes)
        topic_docs = {topic: [] for topic in topics} # Lưu doc_ids thuộc topic
        
        # 2. Lấy các tài liệu đã được gán nhãn thủ công (User-labeled Docs)
        labeled_docs_set = set(classifier.user_labels.keys())
        for doc_id, labels in classifier.user_labels.items():
            for lb in labels:               # duyệt từng nhãn con
                if lb in topic_docs:
                    topic_docs[lb].append(int(doc_id))


        # 3. Phân loại tự động các tài liệu còn lại (Auto-classified Docs)
        # Chỉ chạy nếu có đủ >= 2 classes để huấn luyện bộ phân loại
        if len(topics) >= 2:
            # Điều chỉnh ngưỡng này để kiểm soát độ "tự tin"
            CONFIDENCE_THRESHOLD = 0.9  # Ví dụ: chỉ chấp nhận dự đoán > 80%

            for doc_id in range(num_docs):
                # Bỏ qua tài liệu đã được người dùng gán nhãn thủ công
                if doc_id in labeled_docs_set:
                    continue
                
                # Lấy dự đoán hàng đầu
                preds, _, confidence = classifier.get_predictions(doc_id, top_k=1)
                
                if preds:
                    label, prob = preds[0]
                    # Chỉ gán tự động nếu dự đoán tự tin và là một trong các nhãn đã tạo
                    if label in topic_docs and prob >= CONFIDENCE_THRESHOLD:
                        topic_docs[label].append(int(doc_id))

        # 4. Tính lại số lượng
        topic_counts = {topic: len(docs) for topic, docs in topic_docs.items()}

        # 5. Định dạng lại kết quả (sắp xếp theo số lượng doc giảm dần)
        sorted_topics = sorted(topics, key=lambda t: topic_counts.get(t, 0), reverse=True)

        return jsonify({
            'status': 'success',
            'topics': sorted_topics,
            'topic_counts': topic_counts,
            'topic_docs': topic_docs,
            'total_topics': len(topics)
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/display', methods=['GET'])
def display():
    """Xuất toàn bộ kết quả: topics + predictions"""
    try:
        user_id = request.args.get('user_id')
        
        if user_id not in USERS:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
        classifier = USERS[user_id]['classifier']
        
        # Get all predictions
        predictions = {}
        if len(classifier.classes) >= 2:
            UNCERTAINTY_THRESHOLD = 0.9   # THRESHOLD bạn chỉnh mức này tùy ý
            
            for doc_id in range(len(USERS[user_id]['documents_processed'])):
                preds, _, confidence = classifier.get_predictions(doc_id, top_k=1)
                
                if not preds:
                    continue

                label, prob = preds[0]

                # chỉ hiện tài liệu không chắc chắn
                # LOẠI doc đã gán nhãn
                # Sử dụng confidence (giá trị thứ 3) để so sánh với ngưỡng, đảm bảo đồng nhất
                if confidence < UNCERTAINTY_THRESHOLD and doc_id not in classifier.user_labels:
                    predictions[int(doc_id)] = (label, confidence)

                # if preds:
                #     # FIX: Ensure values are python native types
                #     predictions[int(doc_id)] = (preds[0][0], float(preds[0][1]))
        
        # Metrics
        ground_truth = np.random.randint(0, max(2, len(classifier.classes)), len(USERS[user_id]['documents_processed']))
        purity, ari, nmi = classifier.compute_metrics(ground_truth)
        
        return jsonify({
            'status': 'success',
            'labeled_count': USERS[user_id]['labeled_count'],
            'topics': list(classifier.classes),
            'topic_count': len(classifier.classes),
            'predictions': predictions,
            # 👇 thêm 2 dòng này
            'user_labels': classifier.user_labels,
            'labeled_docs': [int(i) for i in classifier.user_labels.keys()],
            'metrics': {
                'purity': purity,
                'ari': ari,
                'nmi': nmi
            }
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/get_topic_summary', methods=['GET'])
def get_topic_summary():
    user_id = request.args.get('user_id')
    label = request.args.get('label')

    if user_id not in USERS:
        return jsonify({"status": "error", "message": "User not found"}), 404

    classifier = USERS[user_id]['classifier']
    docs = USERS[user_id]['documents_raw']

    # list docs in topic
    labels_map = getattr(classifier, 'user_labels', {}) 
    
    # FIX: Explicitly cast to int for JSON compatibility
    doc_ids = [int(i) for i, lab in labels_map.items() if lab == label]
    
    texts = [docs[i] if isinstance(docs[i], str) else docs[i].get("text","") for i in doc_ids]

    summary = llm_suggester.generate_label_summary(label, texts)

    return jsonify({
        "status": "success",
        "label": label,
        "count": len(doc_ids),
        "summary": summary,
        "doc_ids": doc_ids
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)