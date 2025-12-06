from datasets import load_dataset

dataset = load_dataset("zli12321/Bills")
train = dataset["train"]
test = dataset["test"]
print(train[0]["text"])
print(train[0]["summary"])
print(train[0]["topic"])