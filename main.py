from fastapi import FastAPI, UploadFile, File

app = FastAPI()


# JPG
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type != "image/jpeg":
        return {"error": "Only JPG allowed"}

    contents = await file.read()

    with open(f"images/{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename}


# audio
@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    contents = await file.read()

    with open(f"audio/{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename}


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
