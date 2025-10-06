FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requisitos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la app
COPY . .

# Render usará esta variable para el puerto
ENV PORT=5000

EXPOSE 5000

CMD ["python", "app.py"]
