import requests
import json
import time

URL = "http://localhost:5000"
BILLS_PATH = "./bills.json"

# LABEL_COUNTS = [10, 20, 50, 100, 200, 500]   # số nhãn muốn thử
LABEL_COUNTS = [50,100,200,500,1000]


# =============================
# Load bills
# =============================
print("Loading bills.json...")

with open(BILLS_PATH, "r", encoding="utf8") as f:
    bills = json.load(f)

gold_labels = {idx: bill["label"] for idx, bill in enumerate(bills)}
total_docs = len(gold_labels)

print("Loaded", total_docs, "documents.\n")
def run_experiment(num_labels: int):

    print("=" * 60)
    print(f"🚀 START EXPERIMENT — {num_labels} MANUAL LABELS")
    print("=" * 60)

    # Tạo user mới
    r = requests.post(f"{URL}/create_user").json()
    user_id = r["user_id"]
    print(f"🆔 New user created: {user_id}")

    al_count = 0        # số lần AL chọn doc thành công
    random_count = 0    # số lần fallback random

    labeled_so_far = 0

    for i in range(num_labels):

        # ----------------------------------------
        # 1) GỌI AL – GET_RECOMMENDED_DOCUMENT
        # ----------------------------------------
        doc = requests.get(
            f"{URL}/get_recommended_document",
            params={"user_id": user_id}
        ).json()

        # Nếu không có doc_id → backend gặp lỗi → fallback RANDOM
        if "doc_id" not in doc:

            print(f"⚠️  [AL FAIL] No doc_id returned at label {i+1}. Backend msg: {doc}")

            # Lấy list tài liệu chưa gán
            disp = requests.get(
                f"{URL}/display",
                params={"user_id": user_id}
            ).json()

            already = set(int(k) for k in disp.get("user_labels", {}).keys())
            all_ids = set(range(total_docs))
            unlabeled = list(all_ids - already)

            if len(unlabeled) == 0:
                print("🛑 No more unlabeled documents. Stopping early.")
                break

            # Random fallback
            doc_id = unlabeled[0]
            label = gold_labels[doc_id]

            random_count += 1
            print(f"👉 [RANDOM] Using doc {doc_id} (label={label})")

        else:
            # AL chọn được doc hợp lệ
            doc_id = doc["doc_id"]
            label = gold_labels[doc_id]
            al_count += 1

            print(f"🧠 [AL] Selected doc {doc_id} with true label '{label}'")

        # ----------------------------------------
        # 2) Gửi nhãn thật vào backend
        # ----------------------------------------
        requests.post(
            f"{URL}/label_document",
            json={
                "user_id": user_id,
                "doc_id": doc_id,
                "label": label
            }
        )

        labeled_so_far += 1

        if labeled_so_far % 10 == 0:
            print(f"  ⏳ Progress: {labeled_so_far}/{num_labels}")

    # ----------------------------------------
    # 3) GET USER LABELS
    # ----------------------------------------
    print("\n📥 Fetching manual labels summary...")
    display_resp = requests.get(
        f"{URL}/display",
        params={"user_id": user_id}
    ).json()

    manual_dict = display_resp.get("user_labels", {})
    manual_ids = set(int(k) for k in manual_dict.keys())
    print(f"✔ {len(manual_ids)} manual labels collected.")

    # ----------------------------------------
    # 4) GET AUTO ASSIGN
    # ----------------------------------------
    topic_resp = requests.get(
        f"{URL}/get_topic_list",
        params={"user_id": user_id}
    ).json()

    topic_docs = topic_resp["topic_docs"]

    auto_list = []
    for pred_label, docs in topic_docs.items():
        for d in docs:
            if d not in manual_ids:
                auto_list.append((d, pred_label))

    # ----------------------------------------
    # 5) COMPUTE PRECISION + COVERAGE
    # ----------------------------------------
    if len(auto_list) == 0:
        precision = 0
    else:
        correct = sum(1 for (d, pred) in auto_list if pred == gold_labels[d])
        precision = correct / len(auto_list)

    remaining = total_docs - labeled_so_far
    coverage = len(auto_list) / remaining if remaining > 0 else 0

    # ----------------------------------------
    # 6) PRINT SUMMARY WITH LOGS
    # ----------------------------------------
    print("\n========== EXPERIMENT SUMMARY ==========")
    print(f"🔢 Total labels requested: {num_labels}")
    print(f"🏷️ Actual labels done: {labeled_so_far}")
    print(f"🤖 Active Learning selections: {al_count}")
    print(f"🎲 Random fallback selections: {random_count}")
    print(f"📌 Auto-assign count: {len(auto_list)}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"📡 Coverage: {coverage:.4f}")
    print("========================================\n")

    return {
        "labels": num_labels,
        "precision": precision,
        "coverage": coverage,
        "auto_assign": len(auto_list),
        "al_count": al_count,
        "random_count": random_count,
        "manual_done": labeled_so_far
    }

# =============================
# Run all experiments
# =============================

results = []

for n in LABEL_COUNTS:
    time.sleep(0.5)
    res = run_experiment(n)
    results.append(res)

# =============================
# Print final table
# =============================
print("\n========== SUMMARY ==========")
print(f"{'Labels':>8} | {'Precision':>10} | {'Coverage':>10} | Auto-assign")
print("-" * 55)

for r in results:
    print(f"{r['labels']:>8} | {r['precision']:.4f}     | {r['coverage']:.4f}     | {r['auto_assign']}")

print("============================\n")