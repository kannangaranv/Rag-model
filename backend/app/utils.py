from uuid import uuid4
from pathlib import Path
from langchain_core.documents import Document
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from app.config import engine, Base
from app.db_utils import load_session_history, save_message
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from app.config import (
    vector_store,
    llm,
    embeddings
)

VECTOR_DIR = Path("vector_store")

vector_db: Optional[FAISS] = None

# Create text chunks from a larger text body
def create_chunks_from_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

# Create document objects from text chunks
def create_documents_from_chunks(chunks, doc_id):
    documents = []
    for chunk in chunks:
        document = Document(
            page_content=chunk,
            metadata={"doc_id": doc_id}
        )
        documents.append(document)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    return documents, uuids

# Delete documents from the vector store
def delete_documents_from_vector_store(doc_id):
    try:
        global vector_db
        if vector_db is None:
            print("Vector store is not loaded.")
            return
        all_docs = vector_db.docstore._dict
        del_list=[]
        for key,doc in all_docs.items():
            if doc.metadata["doc_id"]==doc_id:
                del_list.append(key)

        if del_list:
            vector_db.delete(ids=del_list)
            vector_db.save_local("vector_store")
            load_vector_store()
            print(f"Deleted {len(del_list)} documents associated with {doc_id} from the vector store.")
        else:
            print(f"No documents found for {doc_id} in the vector store.")
    except Exception as e:
        print(f"Error deleting documents from vector store: {e}")

# Upload documents to the vector store
def upload_documents_to_vector_store(documents, uuids):
    global vector_db
    if (VECTOR_DIR / "index.faiss").exists() and (VECTOR_DIR / "index.pkl").exists():
        vector_db.add_documents(documents=documents, ids=uuids)
        vector_db.save_local("vector_store")
        load_vector_store()
    else:
        vector_store.add_documents(documents=documents, ids=uuids)
        vector_store.save_local("vector_store")
        load_vector_store()
    print("Documents uploaded to vector store successfully.")

# Create the RAG chain
def create_rag_chain():
    global vector_db
    if vector_db is None:
        raise ValueError("Vector store is not loaded. Please upload documents first.")

    retriever = vector_db.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
        )
    
    qa_system_prompt = """You are a helpful assistant for the BoardPAC application, powered by GPT-4o and Retrieval-Augmented Generation (RAG). 
All relevant BoardPAC knowledge is stored in the knowledge base, and you should answer based only on the provided retrieved context. Retrieved Context (Top Relevant Chunks): {context}

Internally follow these steps:
1. Summarize the user question in simpler words.
2. Identify which retrieved text chunks from the provided context are directly relevant to the question.
3. Combine those chunks into a clear outline.
4. Draft a single, coherent, complete answer using only the relevant chunks.

Output Rules:
- **Only** return the final refined answer.
- **Always** format the answer as fully valid HTML — using headings (`<h2>`), paragraphs (`<p>`), ordered/unordered lists (`<ol>`/`<ul>`), list items (`<li>`), and bold (`<strong>`) where needed.
- **Do not** return Markdown or plain text or html as a string."""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        load_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

# Invoke the RAG chain and save the messages
def invoke_and_save(session_id, input_text):
    
    save_message(session_id, "human", input_text)
    conversational_rag_chain = create_rag_chain()
    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    save_message(session_id, "ai", result)
    return result

# def get_similarity_context(query, k=6,):
#     query_embedding = embeddings.embed_query(query)
#     results = vector_db.similarity_search_with_score_by_vector(query_embedding, k=k)
#     retrieved_docs = [doc.page_content for doc, _ in results]
#     context = "\n\n".join(retrieved_docs)
#     return retrieved_docs

# def get_llm_response(query, context):
#     messages = [
#         {
#             "role": "system",
#             "content": """You are a helpful assistant for the BoardPAC application, powered by GPT-4o and Retrieval-Augmented Generation (RAG). 
# All relevant BoardPAC knowledge is stored in the knowledge base, and you should answer based only on the provided retrieved context.

# Internally follow these steps:
# 1. Summarize the user question in simpler words.
# 2. Identify which retrieved text chunks from the provided context are directly relevant to the question.
# 3. Combine those chunks into a clear outline.
# 4. Draft a single, coherent, complete answer using only the relevant chunks.

# Output Rules:
# - **Only** return the final refined answer.
# - **Always** format the answer as fully valid HTML — using headings (`<h2>`), paragraphs (`<p>`), ordered/unordered lists (`<ol>`/`<ul>`), list items (`<li>`), and bold (`<strong>`) where needed.
# - **Do not** return Markdown or plain text or html as a string."""
#         },
#         {
#             "role": "user",
#             "content": f"""User Query:
# {query}

# Retrieved Context (Top Relevant Chunks):
# {context}"""
#         }
#     ]
#     response = llm.invoke(messages)
#     return response.content

# Load the vector store from disk
def load_vector_store() -> None:
    global vector_db
    if (VECTOR_DIR / "index.faiss").exists() and (VECTOR_DIR / "index.pkl").exists():
        vector_db = FAISS.load_local(
            folder_path=str(VECTOR_DIR),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"VectorStore Loaded from {VECTOR_DIR}")
    else:
        vector_db = None
        print("VectorStore Not found; start by uploading a PDF/Video.")

