FROM python:3.10-slim
WORKDIR /app
COPY api.py .
COPY requirements.txt .
RUN pip3 install -r requirements.txt
# Streamlit tourne par défaut sur le port 8501 (ou 8081 selon votre config compose)
CMD streamlit run api.py --server.port 8081 --server.address 0.0.0.0
