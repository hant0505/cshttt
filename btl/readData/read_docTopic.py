with open("./model_LDA/MalletLda/modelFiles/doc-topics.txt", "r", encoding="utf8") as f:
    for i in range(2):
        line = f.readline()
        print(f"{i}: {line.rstrip()}")

count = sum(1 for _ in open("model_LDA/MalletLda/modelFiles/doc-topics.txt", "r", encoding="utf8"))

print("Total lines:", count)
print("Documents (excluding header) =", count - 1)
