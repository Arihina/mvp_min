Install Tesseract OCR
```
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-rus
```
Install ffmpeg
```
sudo apt install ffmpeg
```
Install Ollama
```
curl -fsSl https://ollama.com/install.sh | sh
```
Install Gemma
```
ollama pull gemma2:2b
```
Install dependencies
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

```

SSL certificate generation
```
openssl req -x509 -newkey rsa:4096 \
-keyout key.pem \
-out cert.pem \
-days 365 \
-nodes
```

Run server:
```python3 main.py```

Swagger UI (docs) http://127.0.0.1:8000/docs/
