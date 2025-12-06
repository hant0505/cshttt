## lưu tạm
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
texts = load_documents()
tfidf_matrix = compute_tfidf(texts)
BASE_CLASSIFIER = EnhancedActiveLearning(doc_topics, tfidf_matrix, combine_weight=0.5)
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
            'documents': texts,
            'labeled_count': 0,
            'labeled_history': [],  # Track labeled docs
            'skipped': set(),       # Track skipped docs
            'current_doc_id': None
        }
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'total_documents': len(texts)
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_recommended_document', methods=['GET'])
def get_recommended_document():
    """Lấy doc tiếp theo (Active Learning)
        Nếu ko chắc chắn ko tự tin thì đưa cho user gán nhãn
        ?Là next button
    """
    try:
        user_id = request.args.get('user_id')
        if user_id not in USERS:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
        classifier = USERS[user_id]['classifier']
        documents = USERS[user_id]['documents']
        
        doc_id, entropy = classifier.recommend_document()
        
        # #skip- Nếu doc mà hệ thống đề xuất nằm trong danh sách user đã skip → tìm doc khác.
        # while doc_id in USERS[user_id]['skipped']:
        #     doc_id, entropy = classifier.recommend_document()

        # Store current doc
        USERS[user_id]['current_doc_id'] = doc_id
        
        doc_text = documents[doc_id] if isinstance(documents[doc_id], str) else documents[doc_id].get('text', '')
        
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
        
        doc_text = USERS[user_id]['documents'][doc_id]
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
            'text': doc_text[:100],  # First 500 chars ***sửa sau á
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

        # 🔥 update current doc here
        USERS[user_id]['current_doc_id'] = doc_id
        
        return jsonify({
            'status': 'success',
            'message': f'Labeled doc {doc_id} as {label}',
            'labeled_count': USERS[user_id]['labeled_count'],
            'current_doc_id': doc_id
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/label_document', methods=['POST'])
# def label_document():
#     """Lưu label cho document"""
#     try:
#         data = request.get_json()
#         user_id = data.get('user_id')
#         doc_id = int(data.get('doc_id'))
#         label = data.get('label')
        
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         classifier = USERS[user_id]['classifier']
#         classifier.label_document(doc_id, label)
#         USERS[user_id]['labeled_count'] += 1
#         USERS[user_id]['labeled_history'].append({
#             'doc_id': doc_id,
#             'label': label,
#             'timestamp': datetime.now().isoformat()
#         })
        
#         return jsonify({
#             'status': 'success',
#             'message': f'Labeled doc {doc_id} as {label}',
#             'labeled_count': USERS[user_id]['labeled_count']
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


""" ẨN TẠM
"""
# @app.route('/recommend_document', methods=['POST'])
# def recommend_document():
#     """Label document + get next recommended (2-in-1)"""
#     try:
#         data = request.get_json()
#         user_id = data.get('user_id')
#         doc_id = int(data.get('doc_id'))
#         label = data.get('label')
        
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         # Label current doc
#         classifier = USERS[user_id]['classifier']
#         classifier.label_document(doc_id, label)
#         USERS[user_id]['labeled_count'] += 1
        
#         # Get next doc
#         next_doc_id, entropy = classifier.recommend_document()

#         while next_doc_id in USERS[user_id]['skipped']:
#             next_doc_id, entropy = classifier.recommend_document()
#         USERS[user_id]['current_doc_id'] = next_doc_id
        
#         doc_text = USERS[user_id]['documents'][next_doc_id]
#         if isinstance(doc_text, dict):
#             doc_text = doc_text.get('text', '')
        
#         return jsonify({
#             'status': 'success',
#             'labeled': doc_id,
#             'next_doc_id': next_doc_id,
#             'next_text': doc_text,
#             'entropy_score': float(entropy) if entropy != -1 else None,
#             'labeled_count': USERS[user_id]['labeled_count']
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


""" hide tạm
"""
# @app.route('/skip_document', methods=['POST'])
# def skip_document():
#     """Skip document hiện tại"""
#     try:
#         data = request.get_json()
#         user_id = data.get('user_id')
#         doc_id = int(data.get('doc_id'))
        
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         USERS[user_id]['skipped'].add(doc_id)
        
#         return jsonify({
#             'status': 'success',
#             'message': f'Skipped doc {doc_id}',
#             'skipped_count': len(USERS[user_id]['skipped'])
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500

"""
"""

# @app.route('/back_document', methods=['POST'])
# def back_document():
#     """Quay lại document trước"""
#     try:
#         data = request.get_json()
#         user_id = data.get('user_id')
        
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         history = USERS[user_id]['labeled_history']
#         if len(history) == 0:
#             return jsonify({'status': 'error', 'message': 'No history'}), 400
        
#         # Pop last label
#         last = history.pop()
#         doc_id = last['doc_id']
        
#         # Rebuild classifier without this label
#         # (Đơn giản: reinstantiate + relabel lại)
#         classifier = USERS[user_id]['classifier']
#         # Ideally: rebuild từ labeled_history
        
#         USERS[user_id]['labeled_count'] -= 1
        
#         return jsonify({
#             'status': 'success',
#             'message': f'Reverted label for doc {doc_id}',
#             'labeled_count': USERS[user_id]['labeled_count']
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500
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
    doc_text = USERS[user_id]['documents'][doc_id]
    if isinstance(doc_text, dict):
        doc_text = doc_text.get("text", "")

    return jsonify({
        "status": "success",
        "doc_id": doc_id,
        "text": doc_text
    })

@app.route('/search', methods=['POST'])
def search_documents():
    data = request.get_json()
    user_id = data.get('user_id')
    query = data.get('query')

    classifier = USERS[user_id]['classifier']
    _documents = USERS[user_id]['documents']

    # compute TF-IDF vector from query
    scores = classifier.semantic_search(query, top_k=50)

    # output format:
    return jsonify({
        'status': 'success',
        'results': [
            {'id': doc_id, 'score': float(score)}
            for doc_id, score in scores
        ]
    })

@app.route('/documents', methods=['GET'])
def get_documents():
    user_id = request.args.get('user_id')

    if user_id is None or user_id not in USERS:
        return jsonify({
            "status": "error",
            "message": "Invalid or missing user_id"
        }), 400

    documents = USERS[user_id]['documents']

    return jsonify({
        "status": "success",
        "documents": [
            {"id": i, "text": documents[i] if isinstance(documents[i], str) else documents[i].get('text', '')}
            for i in range(len(documents))
        ]
    })

@app.route('/get_topic_list', methods=['GET'])
def get_topic_list():
    """Danh sách các topic đã được user tạo + docs thuộc topic"""
    try:
        user_id = request.args.get('user_id')

        if user_id not in USERS:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
        classifier = USERS[user_id]['classifier']

        # list topic names
        topics = list(classifier.classes)

        # count docs per topic
        topic_counts = {}
        for label in classifier.labeled_y:
            topic_counts[label] = topic_counts.get(label, 0) + 1

        # map topic -> list[doc_id]
        topic_docs = {}
        for doc_id, label in classifier.labeled_docs.items():
            topic_docs.setdefault(label, []).append(doc_id)

        return jsonify({
            'status': 'success',
            'topics': topics,
            'topic_counts': topic_counts,
            'topic_docs': topic_docs,
            'total_topics': len(topics)
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/get_topic_list', methods=['GET'])
# def get_topic_list():
#     """Danh sách topics đã label"""
#     try:
#         user_id = request.args.get('user_id')
        
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         classifier = USERS[user_id]['classifier']
#         topics = list(classifier.classes)
        
#         # Count docs per topic
#         topic_counts = {}
#         for label in classifier.labeled_y:
#             topic_counts[label] = topic_counts.get(label, 0) + 1
        
#         # list topic to doc ids mapping
#         topic_docs = {}
#         for doc_id, label in classifier.labeled_docs.items():
#             topic_docs.setdefault(label, []).append(doc_id)

#         return jsonify({
#             'status': 'success',
#             'topics': topics,
#             'topic_counts': topic_counts,
#             'total_topics': len(topics)
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500

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
            for doc_id in range(len(USERS[user_id]['documents'])):
                preds, _ = classifier.get_predictions(doc_id, top_k=1)
                predictions[doc_id] = preds[0] if preds else None
        
        # Metrics
        ground_truth = np.random.randint(0, max(2, len(classifier.classes)), len(USERS[user_id]['documents']))
        purity, ari, nmi = classifier.compute_metrics(ground_truth)
        print("DEBUG:" + "ari =", ari, "nmi =", nmi, "purity =", purity)
        return jsonify({
            'status': 'success',
            'labeled_count': USERS[user_id]['labeled_count'],
            'topics': list(classifier.classes),
            'topic_count': len(classifier.classes),
            'predictions': predictions, # Có thể quá lớn quá dài nên ko print ra
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
    docs = USERS[user_id]['documents']

    # list docs in topic
    doc_ids = [i for i, lab in classifier.labeled_docs.items() if lab == label]
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
    app.run(debug=True, port=5000)