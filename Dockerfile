FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Gradio: 7860, FastAPI: 8000 portlarını aç
EXPOSE 7860 8000

# FastAPI'yi arka planda, Gradio'yu önde çalıştır
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & python app_gradio.py"]
