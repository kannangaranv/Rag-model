from langchain_community.vectorstores import FAISS
from embedding import embeddings
from llm import llm

vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    query_embedding = embeddings.embed_query(query)

    results = vector_store.similarity_search_with_score_by_vector(
        query_embedding, k=3
    )

    retrieved_docs = [doc.page_content for doc, _ in results]

    context = "\n\n".join(retrieved_docs)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the context provided to answer the user's question accurately and clearly."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]

    response = llm.invoke(messages)
    print("LLM Response:", response.content)
