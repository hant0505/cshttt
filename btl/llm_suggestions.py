"""
LLM Suggestions dùng OpenRouter API + Llama 3.1 8B
- Gọi LLM để suggest labels cho document
- Cache suggestions để tránh gọi API quá nhiều
"""

import requests
import json
from typing import List, Tuple
from functools import lru_cache

class LLMSuggestionsOpenRouter:
    """Generate label suggestions dùng OpenRouter API + Llama 3.1 8B"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.1-8b-instruct"):
        """
        api_key: OpenRouter API key
        model: Model name (default: Llama 3.1 8B Instruct)
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.suggestion_cache = {}  # Cache để tránh gọi API nhiều lần
    
    def suggest_labels_for_doc(self, doc_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Gọi LLM để suggest labels cho một document
        
        Returns: list of (label, confidence) tuples
        """
        # Check cache trước
        doc_hash = hash(doc_text[:100])  # Use first 100 chars as key
        if doc_hash in self.suggestion_cache:
            return self.suggestion_cache[doc_hash]
        
        # Truncate text nếu quá dài (tiết kiệm API cost)
        text_preview = doc_text[:500] if len(doc_text) > 500 else doc_text
        
        # Prompt for LLM
        prompt = f"""Analyze this document and suggest top {top_k} topic labels.

Document: "{text_preview}"

Respond in JSON format ONLY (no other text):
{{
  "labels": [
    {{"label": "Topic Name", "confidence": 0.95}},
    {{"label": "Topic Name", "confidence": 0.80}},
    {{"label": "Topic Name", "confidence": 0.65}}
  ]
}}

Generate realistic, concise labels (1-3 words each).
"""
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 300,
                    "top_p": 0.9
                },
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return self._fallback_suggestions()
            
            # Parse response
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Extract JSON
            try:
                # Remove markdown code blocks nếu có
                if "```" in content:
                    content = content.split("```")[1].replace("json", "").strip()
                
                result = json.loads(content)
                suggestions = [
                    (item['label'], float(item['confidence']))
                    for item in result.get('labels', [])
                ][:top_k]
                
                # Cache result
                self.suggestion_cache[doc_hash] = suggestions
                return suggestions
            except json.JSONDecodeError:
                print(f"JSON Parse Error: {content}")
                return self._fallback_suggestions()
        
        except requests.Timeout:
            print("API Timeout - using fallback")
            return self._fallback_suggestions()
        except Exception as e:
            print(f"API Error: {e}")
            return self._fallback_suggestions()
    
    def _fallback_suggestions(self) -> List[Tuple[str, float]]:
        """Fallback suggestions nếu API call fail"""
        return [
            ("Topic A", 0.7),
            ("Topic B", 0.5),
            ("Topic C", 0.3)
        ]
    
    def generate_label_summary(self, label: str, sample_docs: List[str]) -> str:
        """
        Tạo short description cho một label dùng LLM
        (optional - cache để tránh gọi API nhiều)
        """
        if len(sample_docs) == 0:
            return f"Label: {label}"
        
        # Use first 2 docs để tạo summary
        combined_text = " ".join(sample_docs[:2])[:300]
        
        prompt = f"""Based on these documents, create a brief 1-sentence description of the "{label}" topic.

Documents: {combined_text}

Description (1 sentence, max 15 words):"""
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                    "max_tokens": 50
                },
                timeout=5
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                return f"{label}: {content}"
        except:
            pass
        
        return f"Label: {label}"