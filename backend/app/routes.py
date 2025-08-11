from fastapi import APIRouter, HTTPException, UploadFile, File
import tempfile
import shutil
from pathlib import Path
from pydantic import BaseModel
from app.utils import (
    convert_pdf_to_markdown,
    create_documents_from_md_text,
    upload_documents_to_vector_store,
    get_similarity_context,
    get_llm_response
)

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.get("/hello")
def read_hello():
    return {"message": "Hello, World!"} 

@router.post("/upload-documents")
async def upload_documents(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)
    finally:
        await file.close()

    try:
        md_text = convert_pdf_to_markdown(temp_path)
        documents, uuids = create_documents_from_md_text(md_text)
        upload_documents_to_vector_store(documents, uuids)
        return {"message": "Documents uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_path.unlink(missing_ok=True)

@router.post("/query")
async def query_documents(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    context = get_similarity_context(payload.query)
    response = get_llm_response(payload.query, context)
    return {"response": response}