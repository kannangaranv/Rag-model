import os
import json
from fastapi import (
    APIRouter,
    HTTPException, 
    UploadFile,
    File,
    Depends
)
import tempfile
import shutil
from uuid import uuid4
from pathlib import Path
from sqlalchemy import text
from app.config import SessionLocal
from app.schemas import (
    DocumentMeta,
    DocumentListResponse,
    QueryRequest,
    VideoListResponse,
    VideoMeta,
    UserLevelRequest,
)
from fastapi import Query
from uuid import UUID
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi import Request, Response
from pathlib import Path
import tempfile, os
import time
from app.pdf_utils import convert_pdf_to_markdown
from app.utils import (
    create_chunks_from_text,
    create_documents_from_chunks,
    upload_documents_to_vector_store,
    invoke_and_save,
    delete_documents_from_vector_store
)
from app.video_utils import get_transcription_from_video
from app.file_utils import _parse_range_header
from starlette import status
from sqlalchemy.exc import SQLAlchemyError
from app.security import get_current_user, require_upload_permission


router = APIRouter(dependencies=[Depends(get_current_user)])

# API to query documents and get responses from llm.
@router.post("/query/{level}")
async def query_documents(level: int, payload: QueryRequest):
    if level < 1 or level > 6:
        raise HTTPException(status_code=422, detail="Invalid user level. Must be 1..6")
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        response = invoke_and_save("123", payload.query, level)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload pdf to vector store and sql server
@router.post("/upload-documents/{level}")
async def upload_documents(level: int, file: UploadFile = File(...), _: str = Depends(require_upload_permission())):
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    if level < 1 or level > 6:
        raise HTTPException(status_code=422, detail="Invalid user level. Must be 1..6")
    
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
                        (Id, FileName, ContentType, FileSizeBytes, Content, MdText, Level)
                    VALUES
                        (CONVERT(uniqueidentifier, :Id),
                         :FileName, :ContentType, :FileSizeBytes, :Content, :MdText, :Level)
                """),
                {
                    "Id": doc_id,
                    "FileName": file.filename or "document.pdf",
                    "ContentType": file.content_type,
                    "FileSizeBytes": file_size,
                    "Content": pdf_bytes,
                    "MdText": md_text,
                    "Level": level,
                }
            )
            db.commit()
        chunks = create_chunks_from_text(md_text)
        documents, uuids = create_documents_from_chunks(chunks, doc_id, level)
        upload_documents_to_vector_store(documents, uuids)

        return {
            "message": "Document uploaded to vector store.",
            "document_id": doc_id
        }
    except Exception as e:
        print(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

# Upload video to vector store and sql server
@router.post("/upload-videos/{level}")
async def upload_videos(level: int, file: UploadFile = File(...), _: str = Depends(require_upload_permission())):
    allowed = {"video/mp4", "video/x-m4v", "video/mpeg", "video/quicktime"}
    if file.content_type not in allowed:
        raise HTTPException(400, detail=f"Unsupported content type: {file.content_type}")
    if level < 1 or level > 6:
        raise HTTPException(status_code=422, detail="Invalid user level. Must be 1..6")

    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    temp_path = Path(tmp_name)

    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024) 
                if not chunk:
                    break
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())

        video_bytes = temp_path.read_bytes()
        file_size = temp_path.stat().st_size
        video_id = str(uuid4()) 
        transcription = get_transcription_from_video(str(temp_path))
        with SessionLocal() as db:
            db.execute(
                text("""
                    INSERT INTO dbo.Videos
                        (Id, FileName, ContentType, FileSizeBytes, Content, Transcript, Level)
                    VALUES
                        (CONVERT(uniqueidentifier, :Id),
                         :FileName, :ContentType, :FileSizeBytes, :Content, :Transcript, :Level)
                """),
                {
                    "Id": video_id,
                    "FileName": file.filename or "video.mp4",
                    "ContentType": file.content_type,
                    "FileSizeBytes": file_size,
                    "Content": video_bytes,
                    "Transcript": transcription,
                    "Level": level,
                }
            )
            db.commit()

        chunks = create_chunks_from_text(transcription)
        documents, uuids = create_documents_from_chunks(chunks, video_id, level)
        upload_documents_to_vector_store(documents, uuids)

        return {
            "message": "Video uploaded to vector store.",
            "video_id": video_id
        }

    except Exception as e:
        raise HTTPException(500, detail=f"Upload failed: {e}")
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


# API to list documents
@router.get("/documents", response_model=DocumentListResponse)
def list_documents(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200), _: str = Depends(require_upload_permission())):
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
                    CASE WHEN MdText IS NULL THEN 0 ELSE 1 END AS HasMd,
                    Level
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
            level=getattr(row, 'Level', None),
        )
        for row in rows
    ]

    return DocumentListResponse(items=items, total=total, page=page, page_size=page_size)

# API to download documents
@router.get("/documents/{doc_id}/download")
def download_document(doc_id: UUID):
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

    file_like = BytesIO(row.Content)

    headers = {
        "Content-Disposition": f'attachment; filename="{row.FileName}"'
    }
    return StreamingResponse(
        file_like,
        media_type=row.ContentType or "application/pdf",
        headers=headers
    )

# API to view documents
@router.get("/documents/{doc_id}/view")
def view_document(doc_id: UUID, request: Request):
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
            return Response(content=chunk, status_code=206, media_type=content_type, headers=headers)

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total),
        "Content-Disposition": f'inline; filename="{file_name}"',
    }
    return Response(content=blob, media_type=content_type, headers=headers)

@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, _: str = Depends(require_upload_permission())):
    doc_id = (doc_id or "").strip()
    try:
        uid = UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid document id")

    with SessionLocal() as db:
        try:
            exists = db.execute(
                text("SELECT 1 FROM dbo.Documents WHERE Id = :id"),
                {"id": str(uid)},
            ).scalar()
        except SQLAlchemyError:
            raise HTTPException(status_code=500, detail="Database error while checking document")

        if not exists:
            raise HTTPException(status_code=404, detail="Document not found")

    try:
        print("Deleting document from vector store:", doc_id)
        delete_documents_from_vector_store(doc_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete from vector store")

    with SessionLocal() as db:
        try:
            result = db.execute(
                text("DELETE FROM dbo.Documents WHERE Id = :id"),
                {"id": str(uid)},
            )
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise HTTPException(status_code=500, detail="Database error while deleting document")

        if getattr(result, "rowcount", 0) == 0:
            raise HTTPException(status_code=404, detail="Document not found")

    return Response(status_code=status.HTTP_204_NO_CONTENT)

# API to list videos
@router.get("/videos", response_model=VideoListResponse)
def list_videos(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    """
    Paginated list of videos stored in dbo.Videos.
    """
    offset = (page - 1) * page_size
    with SessionLocal() as db:
        total = db.execute(text("SELECT COUNT(*) AS cnt FROM dbo.Videos")).scalar_one()

        rows = db.execute(
            text("""
                SELECT
                    Id,
                    FileName,
                    ContentType,
                    FileSizeBytes,
                    UploadedAt,
                    Level
                FROM dbo.Videos
                ORDER BY UploadedAt DESC
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
            """),
            {"offset": offset, "limit": page_size}
        ).all()

    items = [
        VideoMeta(
            id=row.Id,
            file_name=row.FileName,
            content_type=row.ContentType,
            file_size_bytes=row.FileSizeBytes,
            uploaded_at=row.UploadedAt,
            level=getattr(row, 'Level', None),
        )
        for row in rows
    ]

    return VideoListResponse(items=items, total=total, page=page, page_size=page_size)

# API to download videos
@router.get("/videos/{video_id}/download")
def download_video(video_id: UUID, _: str = Depends(require_upload_permission())):
    """
    Download the full video as an attachment.
    """
    with SessionLocal() as db:
        row = db.execute(
            text("""
                SELECT FileName, ContentType, Content
                FROM dbo.Videos
                WHERE Id = CONVERT(uniqueidentifier, :id)
            """),
            {"id": str(video_id)}
        ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Video not found")

    file_like = BytesIO(row.Content)
    headers = {
        "Content-Disposition": f'attachment; filename="{row.FileName}"'
    }
    return StreamingResponse(
        file_like,
        media_type=row.ContentType or "video/mp4",
        headers=headers
    )

# API to view videos
@router.get("/videos/{video_id}/view")
def view_video(video_id: UUID, request: Request, _: str = Depends(require_upload_permission())):
    """
    Inline video view with HTTP Range support for efficient streaming/seeking.
    """
    with SessionLocal() as db:
        row = db.execute(
            text("""
                SELECT FileName, ContentType, Content
                FROM dbo.Videos
                WHERE Id = CONVERT(uniqueidentifier, :id)
            """),
            {"id": str(video_id)}
        ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Video not found")

    content_type = row.ContentType or "video/mp4"
    file_name = row.FileName or "video.mp4"
    blob: bytes = row.Content
    total = len(blob)

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
            return Response(
                content=chunk,
                status_code=206,
                media_type=content_type,
                headers=headers
            )

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total),
        "Content-Disposition": f'inline; filename="{file_name}"',
    }
    return Response(content=blob, media_type=content_type, headers=headers)

@router.delete("/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: str, _: str = Depends(require_upload_permission())):
    video_id = (video_id or "").strip()
    try:
        uid = UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid video id")

    with SessionLocal() as db:
        try:
            exists = db.execute(
                text("SELECT 1 FROM dbo.Videos WHERE Id = :id"),
                {"id": str(uid)},
            ).scalar()
        except SQLAlchemyError:
            raise HTTPException(status_code=500, detail="Database error while checking video")

        if not exists:
            raise HTTPException(status_code=404, detail="Video not found")

    try:
       print("Deleting video from vector store:", video_id)
       delete_documents_from_vector_store(video_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete from vector store")

    with SessionLocal() as db:
        try:
            result = db.execute(
                text("DELETE FROM dbo.Videos WHERE Id = :id"),
                {"id": str(uid)},
            )
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise HTTPException(status_code=500, detail="Database error while deleting video")

        if getattr(result, "rowcount", 0) == 0:
            raise HTTPException(status_code=404, detail="Video not found")

    return Response(status_code=status.HTTP_204_NO_CONTENT)
    
