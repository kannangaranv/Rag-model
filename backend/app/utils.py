from uuid import uuid4
from pathlib import Path
from langchain_core.documents import Document
from typing import Optional, List
from langchain_community.vectorstores import FAISS


from app.config import (
    vector_store,
    llm,
    embeddings
)

VECTOR_DIR = Path("vector_store")

vector_db: Optional[FAISS] = None

def create_chunks_from_text(text, chunk_size=500, overlap=50):
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
        print(f"VectorStore Loaded from {VECTOR_DIR}")
    else:
        vector_db = None
        print("VectorStore Not found; start by uploading a PDF.")

