import json
import json

with open("./bills.json", "r", encoding="utf8") as f:
    data = json.load(f)

# # In 10 dòng đầu tiên
# if isinstance(data, list):
#     for i, doc in enumerate(data[:1]):
#         print(f"{i}: {doc}")
# elif isinstance(data, dict):
#     keys = list(data.keys())
#     for i, k in enumerate(keys[:1]):
#         print(f"{i}: {k} -> {data[k]}")
# # else:
# #     print(data)
if isinstance(data, list):
    print("List of length:", len(data))
    print("First 10 types:", [type(doc) for doc in data[:10]])
elif isinstance(data, dict):
    print("Dict of keys:", list(data.keys())[:10])
    print("First 10 values types:", [type(data[k]) for k in list(data.keys())[:10]])


# Lấy document đầu tiên
first_doc = data[0]

print("\nKeys of first document:", first_doc.keys())

# In RAW TEXT
raw_text = first_doc["text"]
print("\nRAW DOCUMENT (first 500 chars):\n")
print(raw_text[:1500])

processed = first_doc["processed_text"]
print("\nPROCESSED TEXT:\n")
print(processed[:1500])

# Nếu có label / topic thì in luôn
if "label" in first_doc:
    print("\nLabel:", first_doc["label"])
if "topic" in first_doc:
    print("\nTopic:", first_doc["topic"])
print(first_doc["sub_labels"])

# Số lượng label trong bills.json
# print("\nTotal documents in bills.json:", len(data['sub_labels']))