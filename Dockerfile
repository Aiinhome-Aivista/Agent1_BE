FROM python:3.11-slim

WORKDIR /app

# System deps for cryptography, mysqlclient, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc default-libmysqlclient-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3004", "--proxy-headers"]
