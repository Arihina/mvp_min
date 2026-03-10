FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1 \
    build-essential \
    cmake \
    libcairo2-dev \
    libjpeg-dev \
    libtiff-dev \
    libgirepository1.0-dev \
    libffi-dev \
    pkg-config \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p audio context

EXPOSE 8443

CMD ["python", "main.py"]