import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#load paper abstracts from JSON
with open("data/medical_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
documents = [item["context"] for item in data] #contents of documents

embedding_model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb') #initialize the embedding model
index = faiss.IndexFlatL2(768) #create a FAISS index for 768-dimensional vectors

#embed the documents and add their embeddings to the FAISS index
document_embeddings = embedding_model.encode(documents, show_progress_bar=True)
index.add(np.array(document_embeddings).astype(np.float32))

def search_documents(query, k=5):
    query_vector = embedding_model.encode([query]) #embed the query text
    D, I = index.search(np.array(query_vector).astype(np.float32), k) #find k nearest neighbors of the query embedding
    return [{"content": documents[i], "source": f"Doc_{i}"} for i in I[0]]
