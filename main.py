import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import pytesseract
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import ollama
import faiss


model = WhisperModel("base", device="cpu")

context_dir = Path("./context")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def process_document(file_path, text_splitter, embed_model):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text_splitter.split_text(text)
    results = []

    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(
            chunk, convert_to_numpy=True, normalize_embeddings=True)
        results.append({
            "content": chunk,
            "source": str(file_path),
            "chunk_id": i,
            "embedding": embedding
        })
    return results


documents = []
metadatas = []
all_embeddings = []

for file in context_dir.glob("*.txt"):
    processed_chunks = process_document(file, text_splitter, embed_model)

    for chunk_data in processed_chunks:
        documents.append(chunk_data["content"])
        metadatas.append({
            "source": chunk_data["source"],
            "chunk_id": chunk_data["chunk_id"]
        })
        all_embeddings.append(chunk_data["embedding"])

embeddings = np.vstack(all_embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

RETRIEVAL_THRESHOLD = 0.4


app = FastAPI()


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

    response = ollama.chat(
        model='gemma2:2b',
        messages=[
            {"role": "user", "content": text}
        ]
    )

    return {
        "filename": file.filename,
        "text": response['content']
    }
