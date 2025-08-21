import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import urllib

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVER   = os.getenv("SQL_SERVER", "NUWANK-BP") 
DATABASE = os.getenv("SQL_DATABASE", "KnowledgeBase")
USER     = os.getenv("SQL_USER", "sa1")                      
PASSWORD = os.getenv("SQL_PASSWORD", "123")
DRIVER   = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = OpenAIEmbeddings()
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY
)

conn_str = (
    f"DRIVER={{{DRIVER}}};"           
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"UID={USER};PWD={PASSWORD};"
    "Encrypt=Yes;"                    
    "TrustServerCertificate=Yes"      
)
odbc_connect = urllib.parse.quote_plus(conn_str)

engine = create_engine(f"mssql+pyodbc:///?odbc_connect={odbc_connect}", fast_executemany=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

