import os
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
    PaperMeta,
    PaperListResponse,
    QueryRequest,
    QueryResponse,
    VideoListResponse,
    VideoMeta,
    UploadDocumentResponse,
)
from fastapi import Query
from uuid import UUID
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi import Request, Response
from pathlib import Path
import tempfile, os
from app.pdf_utils import convert_pdf_to_markdown
from app.utils import (
    create_chunks_from_text,
    create_documents_from_chunks,
    create_documents_from_vector_sentences,
    upload_documents_to_vector_store,
    upload_papers_to_vector_store,
    invoke_auto_route_and_save,
    delete_documents_from_vector_store,
    delete_papers_from_vector_store,
    upload_manual_profile_to_vector_store,
    upload_paper_profile_to_vector_store,
    delete_manual_profile_from_vector_store,
    delete_paper_profile_from_vector_store,
)
from app.video_utils import get_transcription_from_video
from app.file_utils import _parse_range_header
from app.user_role_utils import excel_to_vector_sentences
from starlette import status
from sqlalchemy.exc import SQLAlchemyError
from app.security import get_current_user, require_upload_permission
from app.models import User
from app.db_utils import upsert_knowledge_profile, delete_knowledge_profile

router = APIRouter()

# API to query documents
@router.post("/query/{level}", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents(level: int, payload: QueryRequest, current_user: User = Depends(get_current_user)):
    username = current_user.Username
    if level < 1 or level > 6:
        raise HTTPException(status_code=422, detail="Invalid user level. Must be 1..6")
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    paper_id = (payload.paper_id or "").strip() or None
    if paper_id:
        with SessionLocal() as db:
            exists = db.execute(
                text("SELECT 1 FROM dbo.Papers WHERE Id = CONVERT(uniqueidentifier, :id)"),
                {"id": paper_id},
            ).scalar()
        if not exists:
            raise HTTPException(status_code=404, detail="Paper not found")
    try:
        response = invoke_auto_route_and_save(
            username,
            payload.query,
            level,
            paper_id=paper_id,
        )
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload document to vector store and sql server
@router.post("/upload-documents/{level}", response_model=UploadDocumentResponse, status_code=status.HTTP_201_CREATED)
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
        profile_text = upload_manual_profile_to_vector_store(
            doc_id=doc_id,
            content=md_text,
            file_name=file.filename,
        )
        upsert_knowledge_profile(
            doc_id=doc_id,
            source_type="manual",
            file_name=file.filename or "document.pdf",
            profile_text=profile_text,
        )

        return UploadDocumentResponse(
            message="Document uploaded to vector store.",
            document_id=doc_id
        )
    except Exception as e:
        print(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

# Upload video to vector store and sql server
@router.post("/upload-videos/{level}", response_model=UploadDocumentResponse, status_code=status.HTTP_201_CREATED)
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
        upload_manual_profile_to_vector_store(
            doc_id=video_id,
            content=transcription,
            file_name=file.filename,
        )

        return UploadDocumentResponse(
            message="Video uploaded to vector store.",
            document_id=video_id
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Upload failed: {e}")
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

# Upload paper to dedicated vector store and sql server
@router.post("/upload-papers/{level}", response_model=UploadDocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_papers(level: int, file: UploadFile = File(...), _: str = Depends(require_upload_permission())):
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
        paper_id = str(uuid4())

        with SessionLocal() as db:
            db.execute(
                text("""
                    INSERT INTO dbo.Papers
                        (Id, FileName, ContentType, FileSizeBytes, Content, MdText, Level)
                    VALUES
                        (CONVERT(uniqueidentifier, :Id),
                         :FileName, :ContentType, :FileSizeBytes, :Content, :MdText, :Level)
                """),
                {
                    "Id": paper_id,
                    "FileName": file.filename or "paper.pdf",
                    "ContentType": file.content_type,
                    "FileSizeBytes": file_size,
                    "Content": pdf_bytes,
                    "MdText": md_text,
                    "Level": level,
                }
            )
            db.commit()

        chunks = create_chunks_from_text(md_text)
        documents, uuids = create_documents_from_chunks(chunks, paper_id, level)
        upload_papers_to_vector_store(documents, uuids)
        profile_text = upload_paper_profile_to_vector_store(
            doc_id=paper_id,
            content=md_text,
            file_name=file.filename,
        )
        upsert_knowledge_profile(
            doc_id=paper_id,
            source_type="paper",
            file_name=file.filename or "paper.pdf",
            profile_text=profile_text,
        )

        return UploadDocumentResponse(
            message="Paper uploaded to dedicated vector store.",
            document_id=paper_id
        )
    except Exception as e:
        print(f"Error uploading paper: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

# @router.post("/papers/{paper_id}/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
# def query_selected_paper(
#     paper_id: UUID,
#     payload: QueryRequest,
#     current_user: User = Depends(get_current_user),
# ):
#     if not payload.query.strip():
#         raise HTTPException(status_code=400, detail="Query cannot be empty.")

#     with SessionLocal() as db:
#         exists = db.execute(
#             text("SELECT 1 FROM dbo.Papers WHERE Id = CONVERT(uniqueidentifier, :id)"),
#             {"id": str(paper_id)},
#         ).scalar()
#     if not exists:
#         raise HTTPException(status_code=404, detail="Paper not found")

#     try:
#         response = invoke_paper_query_and_save(
#             session_id=current_user.Username,
#             input_text=payload.query,
#             paper_id=str(paper_id),
#         )
#         return QueryResponse(response=response)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# API to upload user roles from Excel file
@router.post("/upload-user-roles", response_model=UploadDocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_user_roles(file: UploadFile = File(...), _: str = Depends(require_upload_permission())):
    if file.content_type not in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",):
        raise HTTPException(status_code=400, detail="Only .xlsx Excel files are accepted.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            temp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)
    finally:
        await file.close()

    try:
        doc_id = str(uuid4())
        records = excel_to_vector_sentences(str(temp_path), doc_id=doc_id)
        if not records:
            raise HTTPException(status_code=400, detail="No valid data found in the Excel file.")
        
        with SessionLocal() as db:
            db.execute(
                text("""
                    INSERT INTO dbo.UserRoleFiles
                        (Id, FileName, ContentType, FileSizeBytes, Content)
                    VALUES
                        (CONVERT(uniqueidentifier, :Id),
                         :FileName, :ContentType, :FileSizeBytes, :Content)
                """),
                {
                    "Id": doc_id,
                    "FileName": file.filename or "user_roles.xlsx",
                    "ContentType": file.content_type,
                    "FileSizeBytes": temp_path.stat().st_size,
                    "Content": temp_path.read_bytes(),
                }
            )
            db.commit()

        documents, uuids = create_documents_from_vector_sentences(records)
        upload_documents_to_vector_store(documents, uuids)

        return UploadDocumentResponse(
            message="User roles uploaded to vector store.",
            document_id=doc_id
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading user roles: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    finally:
        if temp_path:
            try:
                temp_path.unlink(missing_ok=True)
            except PermissionError:
                import time
                for _ in range(5):
                    try:
                        time.sleep(0.1)
                        temp_path.unlink(missing_ok=True)
                        break
                    except PermissionError:
                        continue


# API to list documents
@router.get("/documents", response_model=DocumentListResponse, status_code=status.HTTP_200_OK)
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
@router.get("/documents/{doc_id}/download", status_code=status.HTTP_200_OK)
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
@router.get("/documents/{doc_id}/view", status_code=status.HTTP_200_OK)
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

# API to delete documents
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
        delete_manual_profile_from_vector_store(doc_id)
        delete_knowledge_profile(doc_id=doc_id, source_type="manual")
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

# API to list papers
@router.get("/papers", response_model=PaperListResponse, status_code=status.HTTP_200_OK)
def list_papers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(get_current_user)
):
    offset = (page - 1) * page_size
    with SessionLocal() as db:
        total = db.execute(text("SELECT COUNT(*) AS cnt FROM dbo.Papers")).scalar_one()

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
                FROM dbo.Papers
                ORDER BY UploadedAt DESC
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
            """),
            {"offset": offset, "limit": page_size}
        ).all()

    items = [
        PaperMeta(
            id=row.Id,
            file_name=row.FileName,
            content_type=row.ContentType,
            file_size_bytes=row.FileSizeBytes,
            uploaded_at=row.UploadedAt,
            has_md_text=bool(row.HasMd),
            level=getattr(row, "Level", None),
        )
        for row in rows
    ]
    return PaperListResponse(items=items, total=total, page=page, page_size=page_size)

# API to download papers
@router.get("/papers/{paper_id}/download", status_code=status.HTTP_200_OK)
def download_paper(paper_id: UUID):
    with SessionLocal() as db:
        row = db.execute(
            text("""
                SELECT FileName, ContentType, Content
                FROM dbo.Papers
                WHERE Id = CONVERT(uniqueidentifier, :id)
            """),
            {"id": str(paper_id)}
        ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Paper not found")

    file_like = BytesIO(row.Content)
    headers = {
        "Content-Disposition": f'attachment; filename="{row.FileName}"'
    }
    return StreamingResponse(
        file_like,
        media_type=row.ContentType or "application/pdf",
        headers=headers
    )

# API to view papers
@router.get("/papers/{paper_id}/view", status_code=status.HTTP_200_OK)
def view_paper(paper_id: UUID, request: Request):
    with SessionLocal() as db:
        row = db.execute(
            text("""
                SELECT FileName, ContentType, Content
                FROM dbo.Papers
                WHERE Id = CONVERT(uniqueidentifier, :id)
            """),
            {"id": str(paper_id)}
        ).first()

    if not row:
        raise HTTPException(status_code=404, detail="Paper not found")

    content_type = row.ContentType or "application/pdf"
    file_name = row.FileName or "paper.pdf"
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

@router.delete("/papers/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_paper(paper_id: str, _: str = Depends(require_upload_permission())):
    paper_id = (paper_id or "").strip()
    try:
        uid = UUID(paper_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid paper id")

    with SessionLocal() as db:
        try:
            exists = db.execute(
                text("SELECT 1 FROM dbo.Papers WHERE Id = :id"),
                {"id": str(uid)},
            ).scalar()
        except SQLAlchemyError:
            raise HTTPException(status_code=500, detail="Database error while checking paper")

        if not exists:
            raise HTTPException(status_code=404, detail="Paper not found")

    try:
        print("Deleting paper from dedicated vector store:", paper_id)
        delete_papers_from_vector_store(paper_id)
        delete_paper_profile_from_vector_store(paper_id)
        delete_knowledge_profile(doc_id=paper_id, source_type="paper")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete from paper vector store")

    with SessionLocal() as db:
        try:
            result = db.execute(
                text("DELETE FROM dbo.Papers WHERE Id = :id"),
                {"id": str(uid)},
            )
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise HTTPException(status_code=500, detail="Database error while deleting paper")

        if getattr(result, "rowcount", 0) == 0:
            raise HTTPException(status_code=404, detail="Paper not found")

    return Response(status_code=status.HTTP_204_NO_CONTENT)

# API to list videos
@router.get("/videos", response_model=VideoListResponse, status_code=status.HTTP_200_OK)
def list_videos(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: str = Depends(require_upload_permission())
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
@router.get("/videos/{video_id}/download", status_code=status.HTTP_200_OK)
def download_video(video_id: UUID):
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
@router.get("/videos/{video_id}/view", status_code=status.HTTP_200_OK)
def view_video(video_id: UUID, request: Request):
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

# API to delete videos
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
       delete_manual_profile_from_vector_store(video_id)
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

# API to delete user role files
@router.delete("/user-roles/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_role_file(file_id: str, _: str = Depends(require_upload_permission())):
    file_id = (file_id or "").strip()
    try:
        uid = UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid file id")

    with SessionLocal() as db:
        try:
            exists = db.execute(
                text("SELECT 1 FROM dbo.UserRoleFiles WHERE Id = :id"),
                {"id": str(uid)},
            ).scalar()
        except SQLAlchemyError:
            raise HTTPException(status_code=500, detail="Database error while checking user role file")

        if not exists:
            raise HTTPException(status_code=404, detail="User role file not found")

    try:
       print("Deleting user role file from vector store:", file_id)
       delete_documents_from_vector_store(file_id)
       delete_manual_profile_from_vector_store(file_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete from vector store")

    with SessionLocal() as db:
        try:
            result = db.execute(
                text("DELETE FROM dbo.UserRoleFiles WHERE Id = :id"),
                {"id": str(uid)},
            )
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise HTTPException(status_code=500, detail="Database error while deleting user role file")

        if getattr(result, "rowcount", 0) == 0:
            raise HTTPException(status_code=404, detail="User role file not found")

    return Response(status_code=status.HTTP_204_NO_CONTENT)
    
