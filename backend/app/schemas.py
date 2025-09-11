from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import UUID

class QueryRequest(BaseModel):
    query: str

class UserLevelRequest(BaseModel):
    user_level: int  # 1=Admin, 2=Board Admin, 3=Sys Admin, 4=Organizer, 5=Actionee, 6=Invittee
    
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

# Auth schemas
class UserCreate(BaseModel):
    username: str
    password: str
    # Levels: 1=Admin, 2=Board Admin, 3=Sys Admin, 4=Organizer, 5=Actionee, 6=Invittee
    level: int = 6  # 1..6

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

# Admin: list users
class UserMeta(BaseModel):
    id: int
    username: str
    level: int
    created_at: Optional[datetime] = None

class UserListResponse(BaseModel):
    items: List[UserMeta]
