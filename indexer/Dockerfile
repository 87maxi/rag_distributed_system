# Usamos una imagen de Python slim para reducir el tamaño
FROM python:3.10-slim

# Evitar que Python genere archivos .pyc y permitir logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para watchdog y comunicación de red
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
# Necesitarás: qdrant-client, requests, watchdog, langchain-text-splitters
COPY ./requirements.txt .
RUN ls -la
RUN pip install  -r requirements.txt

# Copiar el script del indexador
COPY ./indexer.py .

