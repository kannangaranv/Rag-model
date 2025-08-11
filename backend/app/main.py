from fastapi import FastAPI
from app.routes import router as api_router
from app.utils import load_vector_store
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_vector_store()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/api")