
from sqlalchemy import Column, DateTime, Integer, LargeBinary, NVARCHAR
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.sql import func
from sqlalchemy import  Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.config import Base

class Document(Base):
    __tablename__ = "Documents"
    Id            = Column(UNIQUEIDENTIFIER, primary_key=True)
    FileName      = Column(NVARCHAR(255),   nullable=False)
    ContentType   = Column(NVARCHAR(100),   nullable=False)
    FileSizeBytes = Column(Integer,         nullable=False)
    Content       = Column(LargeBinary,     nullable=False)
    MdText        = Column(NVARCHAR(None))  
    Level         = Column(Integer,         nullable=True)
    UploadedAt    = Column(DateTime(timezone=False), server_default=func.sysutcdatetime())

class Video(Base):
    __tablename__ = "Videos"
    Id            = Column(UNIQUEIDENTIFIER, primary_key=True)
    FileName      = Column(NVARCHAR(255),   nullable=False)
    ContentType   = Column(NVARCHAR(100),   nullable=False)
    FileSizeBytes = Column(Integer,         nullable=False)
    Content       = Column(LargeBinary,     nullable=False)
    Transcript    = Column(NVARCHAR(None))  
    Level         = Column(Integer,         nullable=True)
    UploadedAt    = Column(DateTime(timezone=False), server_default=func.sysutcdatetime())

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False)  # FIXED
    messages = relationship("Message", back_populates="session")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

class User(Base):
    __tablename__ = "Users"
    Id = Column(Integer, primary_key=True, autoincrement=True)
    Username = Column(NVARCHAR(100), unique=True, nullable=False, index=True)
    HashedPassword = Column(NVARCHAR(255), nullable=False)
    Level = Column(Integer, nullable=False, default=1)
    CreatedAt = Column(DateTime(timezone=False), server_default=func.sysutcdatetime())
