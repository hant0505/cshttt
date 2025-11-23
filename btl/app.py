# simplified_bass.py
class SimplifiedBASS:
    """Mimic tác giả's User class, nhưng dùng Enhanced AL + LLM"""
    
    def __init__(self, corpus_path, model_path, llm_api_key):
        """Load data giống tác giả"""
        from init_classifier import load_doc_topics, load_documents, compute_tfidf
        from classifier_enhanced import EnhancedActiveLearning
        from llm_suggestions import LLMSuggestionsOpenRouter
        
        self.doc_topics = load_doc_topics(model_path)
        self.texts = load_documents(corpus_path)
        self.tfidf_matrix = compute_tfidf(self.texts)
        
        self.classifier = EnhancedActiveLearning(
            self.doc_topics, 
            self.tfidf_matrix, 
            combine_weight=0.5
        )
        self.llm = LLMSuggestionsOpenRouter(llm_api_key)
        
        print("✓ SimplifiedBASS initialized")
    
    def get_doc_information(self, doc_id: int):
        """Return: {text, summary, llm_labels}"""
        doc_text = self.texts[doc_id]
        
        # Get LLM suggestions
        suggestions = self.llm.suggest_labels_for_doc(doc_text, top_k=3)
        llm_labels = [label for label, _ in suggestions]
        
        return {
            'doc_id': doc_id,
            'text': doc_text,
            'llm_labels': llm_labels,
            'suggestions': suggestions  # With confidence
        }
    
    def label_document(self, user_label: str, doc_id: int):
        """Label + retrain classifier"""
        self.classifier.label_document(doc_id, user_label)
        print(f"✓ Labeled doc {doc_id} as {user_label}")
    
    def get_recommended_document(self):
        """AL select document"""
        doc_id, entropy = self.classifier.recommend_document()
        return doc_id, entropy