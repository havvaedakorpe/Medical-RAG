from datasets import load_dataset
import csv
import random

#load the dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
dataset = ds["train"]  #use train split

#select 100 random queries
valid_data = [item for item in dataset if item["question"] and item["long_answer"]]
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
print("done")
