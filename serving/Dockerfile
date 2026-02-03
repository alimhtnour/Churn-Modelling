FROM python:3.10-slim
WORKDIR /app
COPY api.py .
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
# Port 8080 comme spécifié dans le projet [cite: 62]
CMD uvicorn api:app --host 0.0.0.0 --port 8080
