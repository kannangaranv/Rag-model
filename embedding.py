from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# embeddings.embed_query("Hello, world!")

# print(embeddings.embed_query("Hello, world!"))