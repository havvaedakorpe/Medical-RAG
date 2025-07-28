import gradio as gr
import requests

API_URL = "http://localhost:8000/query" #local backend api

def query_medical_rag(question):
    try:
        response = requests.post(API_URL, json={"query": question}) #send POST request with the query
        if response.status_code == 200:
            result = response.json()
            sources = "\n".join(result.get("sources", []))
            return f"Answer:\n{result['answer']}\n\n---\nResources:\n{sources}\nRetrieval: {result['retrieval_time']:.2f}s | Generation: {result['generation_time']:.2f}s"
        else:
            return f"Server Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"
#gradio interface:
app = gr.Interface(
    fn=query_medical_rag,
    inputs=gr.Textbox(label="Medical Query"),
    outputs="text",
    title="Medical RAG Demo"
)
app.launch(server_name="0.0.0.0", server_port=7860)
