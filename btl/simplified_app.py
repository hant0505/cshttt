# """
# BASS Simplified Flask API
# - /create_user: tạo user mới
# - /get_recommended_document: lấy doc được recommend
# - /label_document: user label một doc
# - /predict: lấy predictions từ classifier
# - /metrics: tính metrics
# """

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# from datetime import datetime
# import uuid
# from simplified_user import SimplifiedUser

# app = Flask(__name__)
# CORS(app)

# # Global session storage
# USERS = {}

# # # Sample documents (có thể thay đổi)
# # SAMPLE_DOCUMENTS = [
# #     {'id': i, 'text': f'Sample document {i} about topic X'} 
# #     for i in range(20)
# # ]
# SAMPLE_DOCUMENTS = [
#   {'id':0, 'text':"The influenza virus outbreak caused many infections in 2019."},
#   {'id':1, 'text':"Stock market fluctuations are related to inflation and interest rates."},
#   {'id':2, 'text':"Research on virus infection and public health measures to prevent outbreaks."},
#   {'id':3, 'text':"The central bank adjusts interest rates to control inflation."},
#   {'id':4, 'text':"New vaccine development improved immune response to viral infections."},
#   {'id':5, 'text':"An article about cooking and recipes for healthy food."},
#   {'id':6, 'text':"Investors worry about stock market bubbles and inflation fears."},
#   {'id':7, 'text':"A scientific paper on virus transmission dynamics and case studies."},
#   {'id':8, 'text':"Sports news: the local team won the tournament last night."},
#   {'id':9, 'text':"Market analysis shows strong performance in technology stocks."},
#   {'id':10, 'text':"Education policy reforms and classroom strategies for teachers."},
#   {'id':11, 'text':"The museum exhibits historical fashion and textiles."},
#   {'id':12, 'text':"Local outbreak investigation identified infection clusters."},
#   {'id':13, 'text':"Economic measures to tackle inflation and support market growth."},
#   {'id':14, 'text':"An opinion piece on environmental policy and green markets."},
#   {'id':15, 'text':"Tutorial: How to use TF-IDF and cosine similarity for search."},
#   {'id':16, 'text':"Healthcare workers study infection control protocols."},
#   {'id':17, 'text':"Company earnings and stock market reactions after quarterly report."},
#   {'id':18, 'text':"Travel blog discussing markets, food and local culture."},
#   {'id':19, 'text':"Machine learning approaches to detect outbreaks from tweets."}
# ]

# # Generate random embeddings cho sample documents
# SAMPLE_EMBEDDINGS = np.random.randn(len(SAMPLE_DOCUMENTS), 10)


# @app.route('/create_user', methods=['GET'])
# def create_user():
#     """
#     Tạo session user mới
#     """
#     try:
#         user_id = str(uuid.uuid4())[:8]
#         user = SimplifiedUser(
#             user_id,
#             SAMPLE_DOCUMENTS,
#             SAMPLE_EMBEDDINGS
#         )
#         USERS[user_id] = user
        
#         return jsonify({
#             'status': 'success',
#             'user_id': user_id,
#             'total_documents': len(SAMPLE_DOCUMENTS),
#             'message': f'User {user_id} created successfully'
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# @app.route('/get_recommended_document', methods=['GET'])
# def get_recommended_document():
#     """
#     Lấy document được recommended (Active Learning)
#     """
#     try:
#         user_id = request.args.get('user_id')
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         user = USERS[user_id]
#         doc_info = user.get_recommended_document()
        
#         return jsonify({
#             'status': 'success',
#             'data': doc_info
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# @app.route('/label_document', methods=['POST'])
# def label_document():
#     """
#     User label một document
#     """
#     try:
#         data = request.get_json()
#         user_id = data.get('user_id')
#         doc_id = data.get('doc_id')
#         label = data.get('label')
        
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         user = USERS[user_id]
#         result = user.label_document(doc_id, label)
        
#         return jsonify({
#             'status': 'success',
#             'data': result
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# @app.route('/predict', methods=['GET'])
# def predict():
#     """
#     Lấy predictions từ classifier cho một document
#     """
#     try:
#         user_id = request.args.get('user_id')
#         doc_id = int(request.args.get('doc_id'))
        
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         user = USERS[user_id]
#         predictions = user.get_classifier_predictions(doc_id)
        
#         return jsonify({
#             'status': 'success',
#             'data': predictions
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# @app.route('/metrics', methods=['GET'])
# def metrics():
#     """
#     Tính metrics (nếu có ground truth)
#     """
#     try:
#         user_id = request.args.get('user_id')
        
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         # Sample ground truth labels (có thể thay đổi)
#         ground_truth = np.random.randint(0, 3, len(SAMPLE_DOCUMENTS))
        
#         user = USERS[user_id]
#         metrics = user.compute_metrics(ground_truth)
        
#         return jsonify({
#             'status': 'success',
#             'data': metrics
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# @app.route('/session_summary', methods=['GET'])
# def session_summary():
#     """
#     Lấy tóm tắt session user
#     """
#     try:
#         user_id = request.args.get('user_id')
        
#         if user_id not in USERS:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
#         user = USERS[user_id]
#         summary = user.get_session_summary()
        
#         return jsonify({
#             'status': 'success',
#             'data': summary
#         }), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True, port=5000)