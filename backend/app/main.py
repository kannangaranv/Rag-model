from fastapi import FastAPI
from app.routes import router as api_router
from app.utils import load_vector_store
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vector_store()
    yield

app = FastAPI(lifespan=lifespan)

ALLOWED_ORIGINS = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")