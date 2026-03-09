from fastapi import HTTPException
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import pytesseract
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File, Body
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware
import ollama
import faiss


audio_model = WhisperModel("base", device="cpu")

context_dir = Path("./context")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model_name = "UrukHan/t5-russian-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
chat_history = []

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

SMALL_TALK_PATTERNS = [
    "привет",
    "здравств",
    "как тебя",
    "кто ты",
    "кто я",
    "как меня зовут",
    "меня зовут",
    "hello",
    "hi"
]


def is_small_talk(text):
    text = text.lower()
    return any(p in text for p in SMALL_TALK_PATTERNS)


def format_chat_history(history, max_turns=10):
    history = history[-max_turns * 2:]
    formatted = ""
    for role, text in history:
        if role == "user":
            formatted += f"Пользователь: {text}\n"
        else:
            formatted += f"Ассистент: {text}\n"
    return formatted.strip()


def build_general_prompt(history, user_question):
    chat = format_chat_history(history)

    return f"""
    Ты русскоязычный AI ассистент.
    Отвечай ТОЛЬКО на русском языке.
    Это общий вопрос, не связанный с документами.
    Отвечай свободно, как обычный ассистент.
    Источники указывать НЕ НУЖНО.

    Предыдущий диалог:
    {chat}

    Вопрос пользователя:
    {user_question}

    Ответ:
    """.strip()


def build_rag_prompt(retrieved_docs, history, user_question):
    context_blocks = []

    for i, doc in enumerate(retrieved_docs, 1):
        context_blocks.append(
            f"[Источник {i}]\n{doc['text']}"
        )

    context = "\n\n".join(context_blocks)
    chat = format_chat_history(history)

    return f"""
    Ты русскоязычный AI ассистент.
    Отвечай ТОЛЬКО на русском языке.
    Используй ТОЛЬКО ту информацию из контекста,
    которая действительно нужна для ответа.
    Если информация не использовалась — НЕ УПОМИНАЙ источник.
    
    Контекст:
    {context}
    
    Предыдущий диалог:
    {chat}
    
    Вопрос пользователя:
    {user_question}
    
    Ответ:
    """.strip()


def retrieve_docs(query, top_k=3):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, idxs = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(idxs[0]):
        score = float(scores[0][rank])
        if score < RETRIEVAL_THRESHOLD:
            continue

        results.append({
            "text": documents[idx],
            "source": metadatas[idx]["source"],
            "chunk_id": metadatas[idx]["chunk_id"],
            "score": score
        })

    return results


def find_used_sources(answer: str, retrieved_docs):
    used = set()
    answer_lower = answer.lower()

    for doc in retrieved_docs:
        words = doc["text"].lower().split()
        overlap = sum(1 for w in words if w in answer_lower)

        if overlap > 5:
            used.add(doc["source"])

    return used


def rag_ollama_answer(user_question, chat_history):
    if is_small_talk(user_question):
        prompt = build_general_prompt(chat_history, user_question)
    else:
        retrieved = retrieve_docs(user_question, top_k=3)
        if not retrieved or retrieved[0]["score"] < 0.5:
            prompt = build_general_prompt(chat_history, user_question)
        else:
            prompt = build_rag_prompt(retrieved, chat_history, user_question)

    response = ollama.chat(
        model='gemma2:2b',
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response['message']['content'].strip()

    if not is_small_talk(user_question) and retrieved:
        used_sources = find_used_sources(answer, retrieved)
        if used_sources:
            answer += "\n\nИсточники:\n" + \
                "\n".join(f"- {s}" for s in used_sources)

    chat_history.append(("user", user_question))
    chat_history.append(("assistant", answer))

    return answer, chat_history


def summarization(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        # max_length=max_input_length,
        truncation=True
    )

    summary_ids = model.generate(
        **inputs,
        # max_length=max_output_length,
        # num_beams=4,
        # early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


app = FastAPI()

origins = ['http://localhost:5173', 'https://localhost:5173']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Images
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/gif",
    "image/tiff",
    "image/webp"
}


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. "
            f"Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))

        extracted_text = pytesseract.image_to_string(
            image, lang='rus+eng')
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"OCR failed: {str(e)}")

    try:
        response = summarization(extracted_text)

        return {
            "filename": file.filename,
            "extracted_text": extracted_text,
            "answer": response,
            "text_length": len(extracted_text)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Summarization failed: {str(e)}")


# audio
@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    contents = await file.read()

    audio_path = f"audio/{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(contents)

    segments, info = audio_model.transcribe(f"audio/{file.filename}")
    user_text = " ".join(segment.text for segment in segments)

    answer, _ = rag_ollama_answer(user_text, chat_history)

    return {
        "filename": file.filename,
        "transcribed_text": user_text,
        "answer": answer
    }


# txt
@app.post("/upload/text")
async def upload_text(item: dict = Body(...)):
    answer, _ = rag_ollama_answer(item['message'], chat_history)
    return {
        "sources": None,
        "answer": answer
    }
