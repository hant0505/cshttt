import json
from collections import Counter

with open("./bills.json", "r", encoding="utf8") as f:
    data = json.load(f)

# ----------------------------
# 1. Collect all labels
# ----------------------------
all_labels = [doc["label"] for doc in data if "label" in doc]

# ----------------------------
# 2. Collect all sub_labels (string)
# ----------------------------
all_sublabels = []
for doc in data:
    sub = doc.get("sub_labels", None)
    if sub and isinstance(sub, str):   # <-- SỬA Ở ĐÂY
        all_sublabels.append(sub)

# ----------------------------
# 3. Count frequencies
# ----------------------------
label_count = Counter(all_labels)
sublabel_count = Counter(all_sublabels)

# ----------------------------
# 4. Output
# ----------------------------
print("====== LABELS ======")
print("Số lượng label khác nhau:", len(label_count))
print(label_count)

print("\n====== SUB LABELS ======")
print("Số lượng sub-label khác nhau:", len(sublabel_count))
print(sublabel_count)

print("\nTổng số documents:", len(data))

# non_empty = 0
# for i, doc in enumerate(data):
#     if doc["sub_labels"]:
#         print("Found at index:", i, "->", doc["sub_labels"])
#         non_empty += 1
#         if non_empty >= 10:
#             break

# print("Total documents with non-empty sub_labels:", non_empty)
