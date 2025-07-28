from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

#load model and tokenizer

model_name = "microsoft/biogpt"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_answer(query, retrieved_docs):
    if not retrieved_docs:
        #use only the query if no documents are retrieved
        prompt = f"Question: {query}\nAnswer:"
    else:
        #include context from documents
        context = "\n".join([doc["content"] for doc in retrieved_docs])
        prompt = f"Based on the following medical context, provide a clear answer to question: {query}\nContext:\n{context}\nAnswer: "
    #generate response
    response = generator(
        prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9
    )[0]["generated_text"]
    answer = response[len(prompt):].strip() #extract the answer
    return answer

