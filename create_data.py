from datasets import load_dataset
import json
from tqdm import tqdm

#load dataset and take first 1000 abstracts 
dataset = load_dataset("MedRAG/pubmed", split="train[:10000]") 

cleaned_data = []
for item in tqdm(dataset, desc="Loading"):
    #take content of the item
    content = item.get("content")
    if not content:
        continue
    cleaned_data.append({"content": content})

#write to the JSON file
with open("data/medical_data.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
