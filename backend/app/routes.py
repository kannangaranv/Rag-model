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
from app.schemas import DocumentMeta, DocumentListResponse
from fastapi import Query
from uuid import UUID
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi import Request, Response

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
    
@router.get("/documents", response_model=DocumentListResponse)
def list_documents(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200)):
    """
    Returns paginated list of stored PDFs (metadata only).
    """
    offset = (page - 1) * page_size
    with SessionLocal() as db:
        total = db.execute(text("SELECT COUNT(*) AS cnt FROM dbo.Documents")).scalar_one()

        rows = db.execute(
            text("""
                SELECT
                    Id,
                    FileName,
                    ContentType,
                    FileSizeBytes,
                    UploadedAt,
                    CASE WHEN MdText IS NULL THEN 0 ELSE 1 END AS HasMd
                FROM dbo.Documents
                ORDER BY UploadedAt DESC
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
            """),
            {"offset": offset, "limit": page_size}
        ).all()

    items = [
        DocumentMeta(
            id=row.Id,
            file_name=row.FileName,
            content_type=row.ContentType,
            file_size_bytes=row.FileSizeBytes,
            uploaded_at=row.UploadedAt,
            has_md_text=bool(row.HasMd),
        )
        for row in rows
    ]

    return DocumentListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/documents/{doc_id}/download")
def download_document(doc_id: UUID):
    """
    Streams the PDF bytes back to client.
    """
    with SessionLocal() as db:
        row = db.execute(
            text("""
                SELECT FileName, ContentType, Content
                FROM dbo.Documents
                WHERE Id = CONVERT(uniqueidentifier, :id)
            """),
            {"id": str(doc_id)}
        ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    # row.Content is bytes (VARBINARY(MAX))
    file_like = BytesIO(row.Content)

    headers = {
        "Content-Disposition": f'attachment; filename="{row.FileName}"'
    }
    return StreamingResponse(
        file_like,
        media_type=row.ContentType or "application/pdf",
        headers=headers
    )

@router.get("/documents/{doc_id}/view")
def view_document(doc_id: UUID, request: Request):
    """
    Streams the PDF bytes with 'Content-Disposition: inline' and Range header support.
    This lets the frontend display the PDF directly in an <iframe> or PDF viewer.
    """
    with SessionLocal() as db:
        row = db.execute(
            text("""
                SELECT FileName, ContentType, Content
                FROM dbo.Documents
                WHERE Id = CONVERT(uniqueidentifier, :id)
            """),
            {"id": str(doc_id)}
        ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    content_type = row.ContentType or "application/pdf"
    file_name = row.FileName or "document.pdf"
    blob: bytes = row.Content
    total = len(blob)

    # Byte-range support for PDF viewers
    range_header = request.headers.get("range")
    if range_header:
        rng = _parse_range_header(range_header, total)
        if rng:
            start, end = rng
            chunk = blob[start:end + 1]
            headers = {
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(chunk)),
                "Content-Disposition": f'inline; filename="{file_name}"',
            }
            # 206 Partial Content
            return Response(content=chunk, status_code=206, media_type=content_type, headers=headers)

    # No/invalid Range → return full file
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total),
        "Content-Disposition": f'inline; filename="{file_name}"',
    }
    return Response(content=blob, media_type=content_type, headers=headers)


def _parse_range_header(range_header: str, file_size: int):
    """
    Parses a Range header like 'bytes=START-END' and returns (start, end) (inclusive).
    If END is omitted, it means 'to the end'. If START is omitted, it's a suffix range (not implemented here).
    """
    try:
        units, _, rng = range_header.partition("=")
        if units.strip().lower() != "bytes":
            return None
        start_s, _, end_s = rng.partition("-")
        if start_s == "":
            # Suffix ranges (bytes=-500) are uncommon for PDF viewers; return None to ignore
            return None
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1
        if start < 0 or end < start or end >= file_size:
            return None
        return start, end
    except Exception:
        return None