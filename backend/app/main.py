from fastapi import FastAPI
from app.routes import router as api_router
from app.auth import router as auth_router
from app.utils import load_vector_store
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.config import Base, engine, ensure_database
from app import models   
from dotenv import load_dotenv
import os

load_dotenv()   

DATABASE = os.getenv("SQL_DATABASE", "KnowledgeBase")
ALLOWED_ORIGINS = [
    "http://localhost:4200",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_database(DATABASE)
    load_vector_store()
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(api_router, prefix="/api")
