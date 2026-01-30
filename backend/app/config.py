import os
import faiss as _faiss  # Optional: only needed when VECTOR_DB=faiss
import urllib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")      
VECTOR_DB = os.getenv("VECTOR_DB", "faiss").lower()
SERVER   = os.getenv("SQL_SERVER", "NUWANK-BP") 
DATABASE = os.getenv("SQL_DATABASE", "KnowledgeBase")
USER     = os.getenv("SQL_USER", "sa1")                      
PASSWORD = os.getenv("SQL_PASSWORD", "123")
DRIVER   = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")

# Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002", api_key = OPENAI_API_KEY)

# Initialize vector store (conditional on provider)
if VECTOR_DB == "faiss":
    from langchain_community.vectorstores import FAISS  
    from langchain_community.docstore.in_memory import InMemoryDocstore  
    if _faiss is None:
        raise ImportError("FAISS backend selected but faiss is not installed")
    index = _faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
else:
    vector_store = None  # Not used when Pinecone is selected

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY
)

def make_conn_str(database: str) -> str:
    conn_str = (
        f"DRIVER={{{DRIVER}}};"
        f"SERVER={SERVER};"
        f"DATABASE={database};"
        f"UID={USER};PWD={PASSWORD};"
        "Encrypt=Yes;"
        "TrustServerCertificate=Yes;"
    )
    return urllib.parse.quote_plus(conn_str)

def ensure_database(db_name: str):
    master_odbc = make_conn_str("master")
    master_engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={master_odbc}",
        echo=False,
        fast_executemany=True,
    )

    with master_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        exists = conn.execute(
            text("SELECT 1 FROM sys.databases WHERE name = :name"),
            {"name": db_name},
        ).scalar()
        if not exists:
            conn.execute(text(f"CREATE DATABASE [{db_name}]"))

# Initialize database connection
conn_str = make_conn_str(DATABASE)
odbc_connect = urllib.parse.quote_plus(conn_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={odbc_connect}", fast_executemany=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


