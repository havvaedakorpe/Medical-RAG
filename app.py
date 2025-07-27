from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
import time

from rag import search_documents, generate_answer

app = FastAPI()

#request data model
class QueryRequest(BaseModel):
    query: str

#response data model
class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    latency: Optional[float] = None

@app.post("/query", response_model=QueryResponse) #endpoint at "/query" that receives QueryRequest and returns QueryResponse
async def get_medical_answer(request: QueryRequest):
    total_start = time.time()
    retrieval_start = time.time()
    
    retrieved_docs = search_documents(request.query)  #find documents relevant to the query
    retrieval_time = time.time() - retrieval_start

    generation_start = time.time()
    llm_response = generate_answer(request.query, retrieved_docs)  #generate answer with LLM based on query and retrieved documents
    generation_time = time.time() - generation_start

    total_time = time.time() - total_start

    return {
        "answer": llm_response,
        "sources": [doc["source"] for doc in retrieved_docs],
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "latency": total_time
    }
