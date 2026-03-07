import pytesseract
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel


app = FastAPI()
model = WhisperModel("base", device="cpu")


# JPG
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type != "image/jpeg":
        return {"error": "Only JPG allowed"}

    contents = await file.read()

    # with open(f"images/{file.filename}", "wb") as f:
    #     f.write(contents)

    try:
        image = Image.open(io.BytesIO(contents))

        extracted_text = pytesseract.image_to_string(
            image, lang='rus+eng')

        return {
            "filename": file.filename,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text)
        }
    except Exception as e:
        return {"error": f"OCR failed: {str(e)}"}


# audio
@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    contents = await file.read()
    txt = ""

    with open(f"audio/{file.filename}", "wb") as f:
        f.write(contents)

    segments, info = model.transcribe(f"audio/{file.filename}")

    for segment in segments:
        txt += segment.text

    return {"text": txt}


# txt
@app.post("/upload/text")
async def upload_text(file: UploadFile = File(...)):
    if file.content_type != "text/plain":
        return {"error": "Only txt allowed"}

    text = (await file.read()).decode("utf-8")

    return {
        "filename": file.filename,
        "text": text
    }
