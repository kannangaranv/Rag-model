
from sqlalchemy import Column, DateTime, Integer, LargeBinary, NVARCHAR
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.sql import func
from app.config import Base

class Document(Base):
    __tablename__ = "Documents"
    Id            = Column(UNIQUEIDENTIFIER, primary_key=True)
    FileName      = Column(NVARCHAR(255),   nullable=False)
    ContentType   = Column(NVARCHAR(100),   nullable=False)
    FileSizeBytes = Column(Integer,         nullable=False)
    Content       = Column(LargeBinary,     nullable=False)
    MdText        = Column(NVARCHAR(None))  
    UploadedAt    = Column(DateTime(timezone=False), server_default=func.sysutcdatetime())

class Video(Base):
    __tablename__ = "Videos"
    Id            = Column(UNIQUEIDENTIFIER, primary_key=True)
    FileName      = Column(NVARCHAR(255),   nullable=False)
    ContentType   = Column(NVARCHAR(100),   nullable=False)
    FileSizeBytes = Column(Integer,         nullable=False)
    Content       = Column(LargeBinary,     nullable=False)
    Transcript    = Column(NVARCHAR(None))  
    UploadedAt    = Column(DateTime(timezone=False), server_default=func.sysutcdatetime())
