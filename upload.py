
# option 2 ---------------------------------------------------------------------------------

# from embedding import embeddings
# from config import conn
# import json

# cursor = conn.cursor()
# vector = embeddings.embed_query("Hello world")
# cursor.execute("INSERT INTO langchain_vectors (content, embedding) VALUES (?, ?)", ("Hello world", json.dumps(vector)))
# conn.commit()
# conn.close()

# option 3 ---------------------------------------------------------------------------------
from document import (
    documents,
    uuids
)
from config import vector_store
# vector_store.index.reset() 
# vector_store.docstore._dict.clear()  
# vector_store.index_to_docstore_id.clear()  

vector_store.add_documents(documents=documents, ids=uuids)
vector_store.save_local("vector_store")