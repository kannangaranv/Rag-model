from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import UUID

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class UploadDocumentResponse(BaseModel):
    message: str
    document_id: str

class DocumentMeta(BaseModel):
    id: UUID
    file_name: str
    content_type: str
    file_size_bytes: int
    uploaded_at: datetime
    has_md_text: bool
    level: Optional[int] = None

class DocumentListResponse(BaseModel):
    items: List[DocumentMeta]
    total: int
    page: int
    page_size: int

class VideoMeta(BaseModel):
    id: UUID
    file_name: str
    content_type: Optional[str] = None
    file_size_bytes: int
    uploaded_at: datetime
    level: Optional[int] = None

class VideoListResponse(BaseModel):
    items: List[VideoMeta]
    total: int
    page: int
    page_size: int

class PaperMeta(BaseModel):
    id: UUID
    file_name: str
    content_type: str
    file_size_bytes: int
    uploaded_at: datetime
    has_md_text: bool
    level: Optional[int] = None

class PaperListResponse(BaseModel):
    items: List[PaperMeta]
    total: int
    page: int
    page_size: int

class UserCreate(BaseModel):
    username: str
    password: str
    level: int = 1  # 1..6

class UserOut(BaseModel):
    id: int
    username: str
    level: int
    can_upload: bool = False

class LoginRequest(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserMeta(BaseModel):
    id: int
    username: str
    level: int
    created_at: Optional[datetime] = None

class UserListResponse(BaseModel):
    items: List[UserMeta]
