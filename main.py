from langchain_community.vectorstores import FAISS
from embedding import embeddings
vector_store = FAISS.load_local(
    "vector_store", 
    embeddings, 
    allow_dangerous_deserialization=True
)

results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")