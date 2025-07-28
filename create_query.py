from datasets import load_dataset
import csv
import json
import random

#load the dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train[:100]")

#select 100 random queries
valid_data = [item for item in ds if item["question"] and item["long_answer"]]
sampled = random.sample(valid_data, min(100, len(valid_data)))

#write to the csv file
with open("queries2.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Query", "Answer"])
    writer.writeheader()
    for item in sampled:
        writer.writerow({
            "Query": item["question"],
            "Answer": item["long_answer"]
        })
print("queries2.csv written.")

contents = []

for item in ds:
    context_list = item.get("context", {}).get("contexts", [])
    for ctx in context_list:
        contents.append({"context": ctx})

with open("data/medical_data.json", "w", encoding="utf-8") as f:
    json.dump(contents, f, ensure_ascii=False, indent=2)

print("medical_data.json written.")
