FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    netcat-openbsd \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --upgrade pip \
    && python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
#COPY preprocessing-service/src/ ./src/
COPY wait-for-db.sh /usr/local/bin/wait-for-db
RUN chmod +x /usr/local/bin/wait-for-db

EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# Run the application
CMD ["python3.11", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
