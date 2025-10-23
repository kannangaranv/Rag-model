from uuid import uuid4
from pathlib import Path
from langchain_core.documents import Document
from typing import Optional
from langchain_community.vectorstores import FAISS
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
def create_chunks_from_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks


# Create document objects from text chunks
def create_documents_from_chunks(chunks, doc_id, level: int | None = None):
    documents = []
    for chunk in chunks:
        document = Document(
            page_content=chunk,
            metadata={"doc_id": doc_id, "user_level": level}
        )
        documents.append(document)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    return documents, uuids

# Create document objects from vector sentences
def create_documents_from_vector_sentences(sentences):
    documents = []
    for entry in sentences:
        text = entry["text"]
        metadata = entry["metadata"]
        document = Document(
            page_content=text,
            metadata=metadata
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
    try:
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
    except Exception as e:
        print(f"Error uploading documents to vector store: {e}")

# Create the RAG chain
def create_rag_chain(level: int | None = None):
    global vector_db
    if vector_db is None:
        raise ValueError("Vector store is not loaded. Please upload documents first.")

    # If a level is provided, filter retrieval to matching chunks
    search_kwargs = {
        "k": 6
    }
    # if level is not None:
    #     search_kwargs["filter"] = {"user_level": level}
    retriever = vector_db.as_retriever(search_kwargs=search_kwargs)
    
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
    
    qa_system_prompt = """
You are a helpful assistant for the BoardPAC application, powered by GPT-4o and Retrieval-Augmented Generation (RAG).

STRICT SCOPE:
- You must ONLY answer questions about BoardPAC.
- If a user asks something unrelated to BoardPAC, or the retrieved context does not contain the answer, politely decline per the Response Policies below.

Retrieved Context (Top Relevant Chunks): {context}

Internal Steps:
1. Restate the user question in simpler words (internally).
2. Identify which retrieved chunks directly answer the question; ignore everything else.
3. If no chunk is relevant or coverage is insufficient, STOP and return the appropriate policy response (see Response Policies).
4. Combine only the relevant information needed to answer the question — no extra background or explanation.
5. Draft a short, clear, **direct answer focused strictly on the user’s question** using ONLY the retrieved context.

Response Policies:
- Out-of-scope (not about BoardPAC):
  Return:
  <h2>Out of Scope</h2>
  <p>I can only answer questions about the BoardPAC system. Please ask a BoardPAC-related question.</p>

- Insufficient context (about BoardPAC, but not found in {context}):
  Return:
  <h2>Not Enough Information</h2>
  <p>I couldn’t find this in the BoardPAC knowledge base. Please provide more details or try another BoardPAC-specific query.</p>

- No hallucinations: Never invent facts not present in the retrieved context.

Output Rules:
- Keep the answer short, focused, and directly addressing the question (“on the point”).
- Use valid HTML tags (<h2>, <p>, <ul>, <li>, <strong>) — no Markdown or plain text.
- Do not include explanations or background unless directly needed to answer the question.
- Only return the final refined answer.

"""


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
def invoke_and_save(session_id, input_text, level: int | None = None):
    save_message(session_id, "human", input_text)
    conversational_rag_chain = create_rag_chain(level)

    res = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )

    answer = res.get("answer")
    docs = res.get("context", [])  

    for i, d in enumerate(docs, 1):
        print(f"\n=== Retrieved #{i} ===")
        print(f"METADATA: {d.metadata}")
        print(f"CONTENT: {d.page_content}")

    save_message(session_id, "ai", answer)
    return answer


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

