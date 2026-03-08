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

Run server:
```uvicorn main:app --reload```

Swagger UI (docs) http://127.0.0.1:8000/docs/
