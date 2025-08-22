import pymupdf4llm
from uuid import uuid4
from pathlib import Path
from langchain_core.documents import Document
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import tempfile
import shutil
import time

from app.config import (
    vector_store,
    llm,
    embeddings
)

VECTOR_DIR = Path("vector_store")

vector_db: Optional[FAISS] = None

def convert_pdf_to_markdown(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text

def create_chunks_from_md_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

def create_documents_from_chunks(chunks):
    documents = []
    for chunk in chunks:
        document = Document(
            page_content=chunk,
        )
        documents.append(document)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    return documents, uuids

def upload_documents_to_vector_store(documents, uuids):
    global vector_db
    if (VECTOR_DIR / "index.faiss").exists() and (VECTOR_DIR / "index.pkl").exists():
        vector_db.add_documents(documents=documents, ids=uuids)
        vector_db.save_local("vector_store")
        print("Documents uploaded to vector store successfully.")
        load_vector_store()
    else:
        vector_store.add_documents(documents=documents, ids=uuids)
        vector_store.save_local("vector_store")
        load_vector_store()
        print("Documents uploaded to vector store successfully.")

def get_similarity_context(query, k=6,):
    query_embedding = embeddings.embed_query(query)
    results = vector_db.similarity_search_with_score_by_vector(query_embedding, k=k)
    retrieved_docs = [doc.page_content for doc, _ in results]
    context = "\n\n".join(retrieved_docs)
    return context

def get_llm_response(query, context):
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant for the BoardPAC application, powered by GPT-4o and Retrieval-Augmented Generation (RAG). 
All relevant BoardPAC knowledge is stored in the knowledge base, and you should answer based only on the provided retrieved context.

Internally follow these steps:
1. Summarize the user question in simpler words.
2. Identify which retrieved text chunks from the provided context are directly relevant to the question.
3. Combine those chunks into a clear outline.
4. Draft a single, coherent, complete answer using only the relevant chunks.

Output Rules:
- **Only** return the final refined answer.
- **Always** format the answer as fully valid HTML — using headings (`<h2>`), paragraphs (`<p>`), ordered/unordered lists (`<ol>`/`<ul>`), list items (`<li>`), and bold (`<strong>`) where needed.
- **Do not** return Markdown or plain text or html as a string."""
        },
        {
            "role": "user",
            "content": f"""User Query:
{query}

Retrieved Context (Top Relevant Chunks):
{context}"""
        }
    ]
    response = llm.invoke(messages)
    return response.content

def load_vector_store() -> None:
    global vector_db
    if (VECTOR_DIR / "index.faiss").exists() and (VECTOR_DIR / "index.pkl").exists():
        vector_db = FAISS.load_local(
            folder_path=str(VECTOR_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[VectorStore] Loaded from {VECTOR_DIR}")
    else:
        vector_db = None
        print("[VectorStore] Not found; start by uploading a PDF.")

def extract_audio(video_path, audio_path):

    t0 = time.perf_counter()

    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio:
            audio.write_audiofile(audio_path, codec='libmp3lame')
            print(f"[OK] Audio extracted to {audio_path}")
        else:
            print("[WARN] No audio track found in the video.")
        clip.reader.close()
        clip.audio.reader.close_proc()
    except Exception as e:
        print(f"[ERR] Error extracting audio: {e}")
        raise
    finally:
        t1 = time.perf_counter()
        print(f"[TIME] Extract audio: {(t1 - t0):.2f}s")

def get_file_size(file_path):
    return os.path.getsize(file_path)

def split_audio(audio_path, chunk_dir, chunk_size=20 * 1024 * 1024):
    print("Splitting audio into chunks...")
    t0 = time.perf_counter()
    os.makedirs(chunk_dir, exist_ok=True)
    chunks = []

    audio = AudioSegment.from_file(audio_path)
    total_size = get_file_size(audio_path)
    total_duration_ms = len(audio)  
    chunk_duration_ms = int(chunk_size / total_size * total_duration_ms)
    if chunk_duration_ms >= total_duration_ms:
        chunk_duration_ms = total_duration_ms 

    for i in range(0, total_duration_ms, chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        idx = i // chunk_duration_ms
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx:03d}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

    t1 = time.perf_counter()
    print(f"[OK] Generated {len(chunks)} chunks in {(t1 - t0):.2f}s "
          f"(est. chunk duration ~{chunk_duration_ms/1000:.1f}s)")
    return chunks, (t1 - t0)

def transcribe_audio_chunk(audio_path, client: OpenAI):
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    return response.text

def transcribe_audio(audio_path):
    print("Transcribing audio...")
    timings = {
        "split_seconds": 0.0,
        "chunks_count": 0,
        "per_chunk_seconds": [],
        "transcription_total_seconds": 0.0
    }

    chunk_dir = tempfile.mkdtemp(prefix="chunks_")
    chunks = []
    try:
        chunks, split_secs = split_audio(audio_path, chunk_dir)
        timings["split_seconds"] = split_secs
        timings["chunks_count"] = len(chunks)

        client = OpenAI()
        t_trans_total_start = time.perf_counter()

        transcript_parts = []
        for i, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            print(f"[INFO] Transcribing chunk {i+1}/{len(chunks)}: {os.path.basename(chunk)}")
            text = transcribe_audio_chunk(chunk, client)
            t1 = time.perf_counter()
            dt = t1 - t0
            timings["per_chunk_seconds"].append(dt)
            print(f"[TIME] Chunk {i+1}: {dt:.2f}s")
            transcript_parts.append(text)

        t_trans_total_end = time.perf_counter()
        timings["transcription_total_seconds"] = t_trans_total_end - t_trans_total_start

        return " ".join(transcript_parts).strip(), timings

    finally:
        try:
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)
        except Exception as e:
            print(f"[WARN] Could not delete chunk dir: {e}")

def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    return f"{int(m)}m {s:.1f}s"

def get_transcription_from_video(video_path: str | Path):
    video_path = os.fspath(video_path)
    t_total_start = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio_file:
        audio_path = temp_audio_file.name

    extract_secs = 0.0
    try:
        t0 = time.perf_counter()
        extract_audio(video_path, audio_path)
        t1 = time.perf_counter()
        extract_secs = t1 - t0

        file_size_mb = get_file_size(audio_path) / (1024 * 1024)
        print(f"[INFO] Temporary audio size: {file_size_mb:.2f} MB")

        transcription, timings = transcribe_audio(audio_path)

        t_total_end = time.perf_counter()
        end_to_end = t_total_end - t_total_start

        print("\n=== Timing Summary ===")
        print(f"Extract audio:            {format_seconds(extract_secs)}")
        print(f"Split audio:              {format_seconds(timings['split_seconds'])} "
              f"(chunks: {timings['chunks_count']})")
        if timings["per_chunk_seconds"]:
            print("Per-chunk transcription:")
            for i, secs in enumerate(timings["per_chunk_seconds"], start=1):
                print(f"  - Chunk {i:02d}:        {format_seconds(secs)}")
        print(f"Transcription total:      {format_seconds(timings['transcription_total_seconds'])}")
        print(f"End-to-end total:         {format_seconds(end_to_end)}")

        return transcription

    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"[WARN] Could not delete temp audio file: {e}")