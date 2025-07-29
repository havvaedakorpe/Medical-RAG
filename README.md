# Medical RAG — Tıbbi Soru-Cevap Sistemi

Medical RAG, medikal alan sorularına anlamlı ve güvenilir cevaplar üretebilmek için geliştirilmiş bir Retrieval-Augmented Generation (RAG) sistemidir. Kullanıcıdan gelen tıbbi sorgular önce semantik olarak ilgili belgeler FAISS kütüphanesi kullanılarak aranır, ardından büyük dil modeli (LLM) ile doğal ve bağlama uygun yanıtlar oluşturulur.

---

## Özellikler

- **Semantik Belge Arama:** FAISS ile yüksek performanslı vektör tabanlı belge arama  
- **Embedding Modeli:** BioBERT tabanlı tıbbi metinlere özel gömülü (embedding) oluşturma  
- **Büyük Dil Modeli:** Tıbbi alan sorularına cevap üretmek için güçlü LLM entegrasyonu  
- **REST API:** FastAPI kullanılarak API şeklinde servis sağlama  
- **Web Arayüzü:** Gradio ile kullanıcı dostu interaktif arayüz  
- **Konteynerizasyon:** Dockerfile ile kolay kurulum ve dağıtım  
- **Performans Testi:** 100 adet medikal sorgu ile test ve analiz imkanı  

---

## Kurulum ve Kullanım (Yerel)

```bash
# Sanal ortam oluştur ve aktif et
python -m venv venv
source venv/bin/activate    # Windows için: venv\Scripts\activate

# Gerekli paketleri yükle
pip install -r requirements.txt

# API server'ı başlat
uvicorn app:app --reload

# Gradio tabanlı web arayüzünü kullanmak için
python app_gradio.py

---

# Docker Kullanımı
docker build -t medical-rag .
docker run -p 7860:7860 -p 8000:8000 medical-rag
http://localhost:7860 #Uygulamaya Erişim

---

# Performans Testi
python performance.py


