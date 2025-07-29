import gradio as gr
import requests

API_URL = "http://localhost:8000/query" #backend api

#send query to the backend and format the response
def query_medical_rag(question):
    question = question.strip()
    if not question:
        return "Please enter a medical query."
    try:
        #send POST request
        response = requests.post(API_URL, json={"query": question})
        if response.status_code == 200:
            #parse JSON response
            result = response.json()
            sources = "\n".join(result.get("sources", []))
            answer = result.get("answer", "No answer found.")
            retrieval_time = result.get("retrieval_time", 0)
            generation_time = result.get("generation_time", 0)
            
            output_md = f"""### Answer:
{answer}

---

### Resources:
{sources if sources else 'No sources available.'}

_Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s_
"""
            return output_md
        else:
            return f"Server Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

#gradio interface for the frontend
app = gr.Interface(
    fn=query_medical_rag,
    inputs=gr.Textbox(label="Medical Query", lines=4, placeholder="Ask a medical question..."),
    outputs=gr.Markdown(),
    title="Medical RAG Demo",
    description="Enter your medical question below and get answers supported by relevant sources.",
    flagging_mode="never",
)

app.launch(server_name="0.0.0.0", server_port=7860) #launch the app
