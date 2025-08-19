import json
from fastapi import APIRouter, HTTPException, UploadFile, File
import tempfile
import shutil
from uuid import uuid4
from pathlib import Path
from pydantic import BaseModel
from app.utils import (
    convert_pdf_to_markdown,
    create_documents_from_md_text,
    upload_documents_to_vector_store,
    get_similarity_context,
    get_llm_response
)
from sqlalchemy import text
from app.config import SessionLocal

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

# api to upload documents to vector store
@router.post("/upload-documents")
async def upload_documents(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)
    finally:
        await file.close()

    try:
        md_text = convert_pdf_to_markdown(temp_path)
        pdf_bytes = temp_path.read_bytes()
        file_size = temp_path.stat().st_size
        doc_id = str(uuid4()) 
        with SessionLocal() as db:
            db.execute(
                text("""
                    INSERT INTO dbo.Documents
                        (Id, FileName, ContentType, FileSizeBytes, Content, MdText)
                    VALUES
                        (CONVERT(uniqueidentifier, :Id),
                         :FileName, :ContentType, :FileSizeBytes, :Content, :MdText)
                """),
                {
                    "Id": doc_id,
                    "FileName": file.filename or "document.pdf",
                    "ContentType": file.content_type,
                    "FileSizeBytes": file_size,
                    "Content": pdf_bytes,
                    "MdText": "",
                }
            )
            db.commit()
        documents, uuids = create_documents_from_md_text(md_text)
        upload_documents_to_vector_store(documents, uuids)

        return {
            "message": "Document saved to SQL Server and uploaded to vector store.",
            "document_id": doc_id
        }

    except Exception as e:
        print(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

# API to query documents and get responses from llm.
@router.post("/query")
async def query_documents(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        context = get_similarity_context(payload.query)
        response = get_llm_response(payload.query, context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))