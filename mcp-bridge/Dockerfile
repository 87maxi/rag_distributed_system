# Usamos 3.11-slim por su equilibrio entre velocidad y ligereza
FROM python:3.11-slim

# Evita que Python genere archivos .pyc y asegura que los logs salgan en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalamos dependencias del sistema necesarias para compilar librerías y redes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Actualizamos pip e instalamos las librerías del proyecto
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código (main.py y otros)
COPY . .

# Comando para ejecutar el servidor unificado MCP + Indexer
ENTRYPOINT ["python", "main.py"]
