import csv
import time
import torch
from tqdm import tqdm 
from rag.retrieval import search_documents
from rag.generation import generate_answer
import evaluate
from bert_score import score

from transformers import AutoModelForCausalLM, AutoTokenizer

# BioGPT modeli (HuggingFace'den)
perplexity_model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
perplexity_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
perplexity_model.eval()

def calculate_perplexity(text):
    inputs = perplexity_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = perplexity_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()


def run_performance_and_evaluation(input_csv, perf_csv):
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        queries = list(reader)

    results = []
    predictions = []
    references = []

    for item in tqdm(queries, desc="Answering Queries"): 
        query = item['Query']
        reference = item['Answer']
        query_length = item.get('Query_Length', 'unknown')
        # Retrieval
        start_retrieval = time.time()
        retrieved_docs = search_documents(query)
        end_retrieval = time.time()
        retrieval_time_ms = (end_retrieval - start_retrieval) * 1000

        # Generation
        start_generation = time.time()
        model_output = generate_answer(query, retrieved_docs)
        end_generation = time.time()
        generation_time_ms = (end_generation - start_generation) * 1000

        total_time_ms = retrieval_time_ms + generation_time_ms

        results.append({
            "Query": query,
            "Query_Length": query_length,
            "Retrieval_Time_MS": round(retrieval_time_ms),
            "Generation_Time_MS": round(generation_time_ms),
            "Total_Time_MS": round(total_time_ms),
        })
        predictions.append(model_output)
        references.append(reference)

    # Performans sonuçları
    with open(perf_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["Query", "Query_Length", "Retrieval_Time_MS", "Generation_Time_MS", "Total_Time_MS"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Performans results are saved. '{perf_csv}'")

    # Metrik hesaplama
    print("\n--- Otomatik Metrikler ---")
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
    print("BLEU:", bleu_score)

    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=predictions, references=references)
    print("ROUGE:", rouge_score)

    meteor = evaluate.load("meteor")
    meteor_score = meteor.compute(predictions=predictions, references=references)
    print("METEOR:", meteor_score)

    P, R, F1 = score(predictions, references, lang="en", model_type="bert-base-uncased")
    print(f"BERTScore (F1): {F1.mean().item():.4f}")
    
    sample_text = "Hypertension is a common chronic condition in adults."
    ppl = calculate_perplexity(sample_text)
    print(f"Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    run_performance_and_evaluation("queries_new.csv", "performance_results..csv")
