from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import UUID

class QueryRequest(BaseModel):
    query: str
    
class DocumentMeta(BaseModel):
    id: UUID
    file_name: str
    content_type: str
    file_size_bytes: int
    uploaded_at: datetime
    has_md_text: bool

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

class VideoListResponse(BaseModel):
    items: List[VideoMeta]
    total: int
    page: int
    page_size: int
