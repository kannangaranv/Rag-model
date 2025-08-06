from langchain_community.vectorstores import FAISS
from embedding import embeddings
vector_store = FAISS.load_local(
    "vector_store", 
    embeddings, 
    allow_dangerous_deserialization=True
)

while True:
    query = input("Enter your query (or 'exit' to quit): ")
    embedding = embeddings.embed_query(query)
    if query.lower() == 'exit':
        break
    results = vector_store.similarity_search_with_score_by_vector(
        embedding, k=2
    )
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")