import pymupdf4llm
from uuid import uuid4
from pathlib import Path
from langchain_core.documents import Document
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from app.config import vector_store,llm,embeddings

VECTOR_DIR = Path("vector_store")

vector_db: Optional[FAISS] = None

def convert_pdf_to_markdown(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    return md_text

def create_documents_from_md_text(md_text):
    documents = []
    for chunk in md_text:
        document = Document(
            page_content=chunk['text'],
            metadata={"page": chunk['metadata']['page']}
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
    else:
        vector_store.add_documents(documents=documents, ids=uuids)
        vector_store.save_local("vector_store")
        print("Documents uploaded to vector store successfully.")

def get_similarity_context(query, k=3,):
    query_embedding = embeddings.embed_query(query)
    results = vector_db.similarity_search_with_score_by_vector(query_embedding, k=k)
    retrieved_docs = [doc.page_content for doc, _ in results]
    context = "\n\n".join(retrieved_docs)
    return context

def get_llm_response(query, context):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the context provided to answer the user's question accurately and clearly. The response should be in html format."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
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