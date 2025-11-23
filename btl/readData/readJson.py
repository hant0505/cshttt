import json
import json

with open("./bills_preprocessed.json", "r", encoding="utf8") as f:
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

"""
File bills_preprocessed.json của bro là một dict với các key là ID của bill và value là nội dung đã được tiền xử lý của bill đó.
"""

# check cấu trúc processed_text 
"""
--> 👉 Value là STRING, không phải dict.

"""
pt = data["processed_text"]

print("Type of processed_text:", type(pt))
print("Number of documents:", len(pt))

# Lấy một phần tử bất kỳ
first_key = next(iter(pt))
first_value = pt[first_key]

print("Example key:", first_key)
print("Type of value:", type(first_value))

# Nếu là dict thì in các keys của nó
if isinstance(first_value, dict):
    print("Keys inside this dict:", list(first_value.keys()))

# In trước 150 ký tự để xem text
print("Preview:", str(first_value)[:150])