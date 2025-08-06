# #  option 1 ---------------------------------------------------------------------------------

# from langchain_sqlserver import SQLServer_VectorStore
# from langchain_community.vectorstores.utils import DistanceStrategy
# from langchain_huggingface import HuggingFaceEmbeddings

# # Initialize your HuggingFace embedding model
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Define connection string
# _CONNECTION_STRING = (
#     "Driver={ODBC Driver 18 for SQL Server};"
#     "Server=NUWANK-BP;"
#     "Database=test;"
#     "UID=sa1;"
#     "PWD=123;"
#     "Encrypt=no;"
#     "TrustServerCertificate=yes;"
#     "MultipleActiveResultSets=True;"
#     "Connection Timeout=60;"
# )

# # Initialize vector store with corrected embedding length
# vector_store = SQLServer_VectorStore(
#     connection_string=_CONNECTION_STRING,
#     distance_strategy=DistanceStrategy.COSINE,
#     embedding_function=embeddings,
#     embedding_length=768,  # ← Correct for all-mpnet-base-v2
#     table_name="langchain_test_table"
# )

# option 2 ---------------------------------------------------------------------------------

# import pyodbc

# conn = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};Server=NUWANK-BP;Database=test;UID=sa1;PWD=123;Encrypt=no;TrustServerCertificate=yes;")


# option 3 ---------------------------------------------------------------------------------
import faiss
from embedding import embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
