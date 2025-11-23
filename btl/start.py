from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random

# ==========================
#  Dataset (20 documents)
# ==========================
texts = [
    "The government plans to reform the national education curriculum next year.",
    "Teachers demand better training programs and updated teaching materials.",
    "Students are struggling with the new standardized test format.",
    "Online learning platforms have increased accessibility for rural areas.",
    "Universities are adopting AI tools to support personalized learning.",

    "Hospitals are facing shortages of medical supplies and staff.",
    "A new vaccine has been approved after successful clinical trials.",
    "Telemedicine services expand healthcare access for remote communities.",
    "Mental health awareness campaigns are becoming more common worldwide.",
    "Researchers are developing AI-based systems for early disease detection.",

    "Tech companies are investing heavily in artificial intelligence research.",
    "Cybersecurity threats are increasing due to more connected devices.",
    "Blockchain is being explored for secure data sharing.",
    "Self-driving cars are undergoing testing in major cities.",
    "Cloud computing adoption is accelerating among small businesses.",

    "Climate change is causing more frequent and severe weather events.",
    "Governments are promoting renewable energy to reduce carbon emissions.",
    "Plastic waste in the oceans is becoming a global environmental crisis.",
    "Wildlife conservation programs are helping endangered species recover.",
    "Scientists warn that global temperatures may rise above safe limits."
]


# ======================================
#  BASS Lite = TFIDF + LDA + Active Learning
# ======================================
class BASSLite:
    def __init__(self, texts, n_topics=6):
        self.texts = texts
        self.n = len(texts)

        # --- Step 1: TF-IDF ---
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = self.vectorizer.fit_transform(texts)

        # --- Step 2: LDA ---
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda_probs = lda.fit_transform(tfidf)

        # --- Step 3: Combine features ---
        self.X = np.hstack([tfidf.toarray(), lda_probs])

        # Active Learning classifier
        self.clf = SGDClassifier(loss="log_loss")
        self.label_encoder = LabelEncoder()

        # Tracking
        self.labeled_ids = []
        self.labels = []
        self.unlabeled = list(range(self.n))

        self.all_labels = ["education", "healthcare", "technology", "environment"]
        self.label_encoder.fit(self.all_labels)


    # --- Step 4: Select doc using uncertainty sampling ---
    def select_document(self):
        if len(self.labeled_ids) < 2:
            return random.choice(self.unlabeled)

        probs = self.clf.predict_proba(self.X)
        entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)

        # Only consider unlabeled docs
        return max(self.unlabeled, key=lambda i: entropy[i])

    # --- Step 5 + Step 6: user label + retrain ---
    def label_document(self, doc_id, label):
        self.labeled_ids.append(doc_id)
        self.labels.append(label)
        self.unlabeled.remove(doc_id)

        # Need at least 2 classes to train 
        # ✔ Không train khi mới có 1 class
        if len(set(self.labels)) < 2:
            return

        """
        Do SGDClassifier yêu cầu ít nhất hai lớp để huấn luyện mô hình phân loại, 
        hệ thống sẽ chỉ bắt đầu huấn luyện khi người dùng đã cung cấp tối thiểu hai nhãn thuộc hai chủ đề khác nhau. 
        Điều này cũng phù hợp với BASS gốc: active learning không khởi động cho đến khi hệ thống có tín hiệu ban đầu từ người dùng.
        """
        y = self.label_encoder.fit_transform(self.labels)

        # self.clf.partial_fit(self.X[self.labeled_ids], y, classes=y)
            # Always use full class set
        self.clf.partial_fit(
            self.X[self.labeled_ids],
            y,
            classes=self.label_encoder.transform(self.all_labels)
        )

    # --- Step 7: cluster full corpus ---
    def predict_clusters(self):
        if len(self.labeled_ids) < 2:
            return {}

        preds = self.label_encoder.inverse_transform(self.clf.predict(self.X))
        clusters = {}
        for i, p in enumerate(preds):
            clusters.setdefault(p, []).append(i)
        return clusters


# ======================================
#  EXAMPLE USAGE (MAIN DEMO)
# ======================================
if __name__ == "__main__":
    bass = BASSLite(texts)

    print("=== BASS Lite Started ===\n")

    # Loop 5 times (simulate user labeling)
    for step in range(5):
        doc_id = bass.select_document()
        print(f"[Step {step+1}] Please label this document:")
        print("Document:", texts[doc_id])

        # --- For testing: auto-label based on index ---
        # Bro có thể nhập input() để tự label nếu muốn
        if doc_id < 5:
            label = "education"
        elif doc_id < 10:
            label = "healthcare"
        elif doc_id < 15:
            label = "technology"
        else:
            label = "environment"

        print("Auto-label as:", label)
        print()

        bass.label_document(doc_id, label)

    print("\n=== Predicted Topic Clusters ===")
    clusters = bass.predict_clusters()

    for topic, doc_ids in clusters.items():
        print(f"\nTopic: {topic}")
        for i in doc_ids:
            print("-", texts[i][:70])  # show first 70 chars
