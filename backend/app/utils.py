from uuid import uuid4
from pathlib import Path
from typing import Optional
import os

from langchain_core.documents import Document
from app.db_utils import load_session_history, save_message
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

try:
    from langchain_community.vectorstores import FAISS
    import faiss as _faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
except Exception:  
    FAISS = None
    _faiss = None
    InMemoryDocstore = None

try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
except Exception:  
    PineconeVectorStore = None
    Pinecone = None

try:
    from langchain_milvus import Milvus as MilvusVectorStore
except Exception:
    MilvusVectorStore = None

from app.config import (
    llm,
    embeddings,
)

VECTOR_DB_PROVIDER = os.getenv("VECTOR_DB", "faiss").lower()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bp-index")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "bp_collection")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")
PAPER_VECTOR_DB_PROVIDER = os.getenv("PAPER_VECTOR_DB", VECTOR_DB_PROVIDER).lower()
PAPER_PINECONE_INDEX_NAME = os.getenv("PAPER_PINECONE_INDEX_NAME", "bp-paper-index")
PAPER_MILVUS_COLLECTION = os.getenv("PAPER_MILVUS_COLLECTION", "bp_paper_collection")

VECTOR_DIR = Path("vector_store")
PAPER_VECTOR_DIR = Path("paper_vector_store")

vector_db: Optional[object] = None
paper_vector_db: Optional[object] = None

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
            load_vector_store()
            if vector_db is None:
                print("Vector store is not loaded.")
                return

        if VECTOR_DB_PROVIDER == "pinecone":
            vector_db.delete(filter={"doc_id": doc_id})
            print(f"Requested deletion in Pinecone for doc_id={doc_id}.")
        elif VECTOR_DB_PROVIDER == "milvus":
            expr = f'doc_id == "{doc_id}"'
            vector_db.delete(expr=expr)
            print(f"Requested deletion in Milvus for doc_id={doc_id}.")
        else:
            if not hasattr(vector_db, "docstore") or not hasattr(vector_db.docstore, "_dict"):
                print("FAISS docstore not available; nothing to delete.")
                return
            all_docs = vector_db.docstore._dict
            del_list = []
            for key, doc in all_docs.items():
                if doc.metadata.get("doc_id") == doc_id:
                    del_list.append(key)

            if del_list:
                vector_db.delete(ids=del_list)
                vector_db.save_local("vector_store")
                load_vector_store()
                print(
                    f"Deleted {len(del_list)} documents associated with {doc_id} from the FAISS vector store."
                )
            else:
                print(f"No documents found for {doc_id} in the FAISS vector store.")
    except Exception as e:
        print(f"Error deleting documents from vector store: {e}")

# Upload documents to the vector store
def upload_documents_to_vector_store(documents, uuids):
    try:
        global vector_db
        if vector_db is None:
            load_vector_store()
        if vector_db is None:
            raise RuntimeError("Vector store is not initialized. Upload aborted.")

        if VECTOR_DB_PROVIDER == "pinecone":
            # Remote index; no save_local
            vector_db.add_documents(documents=documents, ids=uuids)
        elif VECTOR_DB_PROVIDER == "milvus":
            # Remote Milvus; no save_local
            vector_db.add_documents(documents=documents, ids=uuids)
        else:
            # Local FAISS; persist to disk
            vector_db.add_documents(documents=documents, ids=uuids)
            vector_db.save_local("vector_store")
            load_vector_store()

        print("Documents uploaded to vector store successfully.")
    except Exception as e:
        print(f"Error uploading documents to vector store: {e}")

# Delete papers from the dedicated paper vector store
def delete_papers_from_vector_store(doc_id):
    try:
        global paper_vector_db
        if paper_vector_db is None:
            load_paper_vector_store()
            if paper_vector_db is None:
                print("Paper vector store is not loaded.")
                return

        if PAPER_VECTOR_DB_PROVIDER == "pinecone":
            paper_vector_db.delete(filter={"doc_id": doc_id})
            print(f"Requested paper deletion in Pinecone for doc_id={doc_id}.")
        elif PAPER_VECTOR_DB_PROVIDER == "milvus":
            expr = f'doc_id == "{doc_id}"'
            paper_vector_db.delete(expr=expr)
            print(f"Requested paper deletion in Milvus for doc_id={doc_id}.")
        else:
            if not hasattr(paper_vector_db, "docstore") or not hasattr(paper_vector_db.docstore, "_dict"):
                print("Paper FAISS docstore not available; nothing to delete.")
                return
            all_docs = paper_vector_db.docstore._dict
            del_list = []
            for key, doc in all_docs.items():
                if doc.metadata.get("doc_id") == doc_id:
                    del_list.append(key)

            if del_list:
                paper_vector_db.delete(ids=del_list)
                paper_vector_db.save_local(str(PAPER_VECTOR_DIR))
                load_paper_vector_store()
                print(
                    f"Deleted {len(del_list)} paper chunks associated with {doc_id} from the paper FAISS vector store."
                )
            else:
                print(f"No paper chunks found for {doc_id} in the FAISS paper vector store.")
    except Exception as e:
        print(f"Error deleting papers from vector store: {e}")

# Upload papers to the dedicated paper vector store
def upload_papers_to_vector_store(documents, uuids):
    try:
        global paper_vector_db
        if paper_vector_db is None:
            load_paper_vector_store()
        if paper_vector_db is None:
            raise RuntimeError("Paper vector store is not initialized. Upload aborted.")

        if PAPER_VECTOR_DB_PROVIDER in ("pinecone", "milvus"):
            paper_vector_db.add_documents(documents=documents, ids=uuids)
        else:
            paper_vector_db.add_documents(documents=documents, ids=uuids)
            paper_vector_db.save_local(str(PAPER_VECTOR_DIR))
            load_paper_vector_store()

        print("Papers uploaded to vector store successfully.")
    except Exception as e:
        print(f"Error uploading papers to vector store: {e}")


def retrieve_paper_documents(query: str, paper_id: str, k: int = 6):
    global paper_vector_db
    if paper_vector_db is None:
        load_paper_vector_store()
    if paper_vector_db is None:
        return []

    try:
        if PAPER_VECTOR_DB_PROVIDER == "milvus":
            docs = paper_vector_db.similarity_search(
                query=query,
                k=k,
                expr=f'doc_id == "{paper_id}"',
            )
        elif PAPER_VECTOR_DB_PROVIDER == "pinecone":
            docs = paper_vector_db.similarity_search(
                query=query,
                k=k,
                filter={"doc_id": paper_id},
            )
        else:
            docs = paper_vector_db.similarity_search(
                query=query,
                k=k,
                filter={"doc_id": paper_id},
            )
    except Exception:
        # Fallback path if provider-specific filtering fails.
        docs = paper_vector_db.similarity_search(query=query, k=k)
        docs = [d for d in docs if d.metadata.get("doc_id") == paper_id]

    return docs or []


def invoke_paper_query_and_save(session_id: str, input_text: str, paper_id: str) -> str:
    paper_session_id = f"{session_id}::paper::{paper_id}"
    save_message(paper_session_id, "human", input_text)
    history = load_session_history(
        paper_session_id,
        max_messages=20,
    )

    contextualize_q_system_prompt = """Rewrite the user's latest message into a concise, standalone question.

Use chat history only to resolve references in the latest message.
Ignore low-information or boilerplate messages in history, including:
- Acknowledgments: "ok", "okay", "thanks", "thank you", "noted"
- Apologies/softeners: "sorry", "apologies"
- Generic refusals/outcomes: "Out of Scope", "Not Enough Information", "I couldn't find"
- Auto-responses/placeholders: "processing...", "please wait", "let me check"
- Meta/system chatter: "as an AI", policy blurbs
- Empty/near-empty content: emojis, lone punctuation
- Repeated echoes of prior answers without new facts

Rules:
- If the latest message is already standalone, return it unchanged.
- Preserve entities, product terms, numbers, dates, and constraints.
- Do NOT answer; return only the rewritten question text (no quotes, no commentary).
"""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    rewrite_chain = contextualize_q_prompt | llm | StrOutputParser()
    standalone_question = rewrite_chain.invoke(
        {"input": input_text, "chat_history": history.messages}
    )
    retrieval_query = (standalone_question or "").strip() or input_text
    if standalone_question:
        save_message(paper_session_id, "human_rewritten", standalone_question)

    docs = retrieve_paper_documents(retrieval_query, paper_id, k=6)

    if not docs:
        answer = (
            "<h2>Not Enough Information</h2>"
            "<p>I could not find relevant content in the selected paper. "
            "Please ask a more specific question about this paper.</p>"
        )
        save_message(paper_session_id, "ai", answer)
        return answer

    paper_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You answer only from the selected paper context.

Context from selected paper: {context}

Rules:
- Answer only from this context.
- If the answer is not in the context, return:
  <h2>Not Enough Information</h2>
  <p>The selected paper does not contain enough information to answer that question.</p>
- Keep answers concise and focused.
- Return valid HTML only (<h2>, <p>, <ul>, <li>, <strong>).""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    paper_qa_chain = create_stuff_documents_chain(llm, paper_prompt)
    answer = paper_qa_chain.invoke(
        {
            "input": retrieval_query,
            "context": docs,
            "chat_history": history.messages,
        }
    )
    save_message(paper_session_id, "ai", answer)
    return answer

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
    
    contextualize_q_system_prompt = """Rewrite the user’s latest message into a concise, standalone question.

Use chat history only to resolve references in the latest message.
Ignore low-information or boilerplate messages in history, including:
- Acknowledgments: "ok", "okay", "thanks", "thank you", "noted"
- Apologies/softeners: "sorry", "apologies"
- Generic refusals/outcomes: "Out of Scope", "Not Enough Information", "I couldn't find"
- Auto-responses/placeholders: "processing...", "please wait", "let me check"
- Meta/system chatter: "as an AI", policy blurbs
- Empty/near-empty content: emojis, lone punctuation
- Repeated echoes of prior answers without new facts

Rules:
- If the latest message is already standalone, return it unchanged.
- Preserve entities, product terms, numbers, dates, and constraints.
- Do NOT answer; return only the rewritten question text (no quotes, no commentary).
"""

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
2. Identify and use which retrieved chunks directly answer the question; ignore everything else.
3. If cannot find relevant chunks to the question, STOP and return the appropriate policy response (see Response Policies).
4. Combine only the relevant information needed to answer the question — no extra background or extra explanation.
5. Draft a short, clear, **direct answer focused strictly on the user’s question** using ONLY the retrieved context.

Response Policies:
- Greetings / small talk (e.g., "hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you", "thanks"):
  Return:
  <p>Hello! I can help with BoardPAC questions. What would you like to know?</p>

- Out-of-scope (not about BoardPAC):
  Return:
  <h2>Out of Scope</h2>
  <p>I can only answer questions about the BoardPAC system. Please ask a BoardPAC-related question.</p>

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

    # Build a rewrite chain to produce a standalone question
    rewrite_chain = contextualize_q_prompt | llm | StrOutputParser()

    # Compose a RAG chain that:
    # - Computes standalone question once
    # - Retrieves context using the history-aware retriever
    # - Feeds the standalone question as the human input to the QA prompt
    # - Returns {"answer", "context"} to match existing callers
    base = RunnablePassthrough.assign(
        standalone_question=rewrite_chain,
        context=history_aware_retriever,
    )

    map_to_qa_inputs = RunnableLambda(
        lambda x: {
            "context": x["context"],
            "input": x["standalone_question"],
            "chat_history": x.get("chat_history", []),
        }
    )

    rag_chain = (
        base
        .assign(answer=map_to_qa_inputs | question_answer_chain)
        | RunnableLambda(
            lambda x: {
                "answer": x["answer"],
                "context": x["context"],
                "standalone_question": x.get("standalone_question"),
            }
        )
    )

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
    standalone_q = res.get("standalone_question")
    docs = res.get("context", [])  

    if standalone_q:
        print(f"\n=== Standalone Question ===\n{standalone_q}")
        # Save the reformulated question as a separate message role
        save_message(session_id, "human_rewritten", standalone_q)

    for i, d in enumerate(docs, 1):
        print(f"\n=== Retrieved #{i} ===")
        print(f"METADATA: {d.metadata}")
        print(f"CONTENT: {d.page_content}")

    save_message(session_id, "ai", answer)
    return answer


# Load the vector store from disk
def load_vector_store() -> None:
    """Initialize or connect to the configured vector store.

    - FAISS: load from local files if present; otherwise create an empty in-memory index.
    - Pinecone: connect to remote index by name; no local files involved.
    """
    global vector_db

    if VECTOR_DB_PROVIDER == "pinecone":
        if Pinecone is None or PineconeVectorStore is None:
            print("Pinecone dependencies are not available. Did you install pinecone-client and langchain-pinecone?")
            vector_db = None
            return
        try:
            _pc = Pinecone(api_key=PINECONE_API_KEY)  
            vector_db = PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings,
            )
            print(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
        except Exception as e:  # noqa: BLE001
            print(f"Failed connecting to Pinecone: {e}")
            vector_db = None
        return

    if VECTOR_DB_PROVIDER == "milvus":
        if MilvusVectorStore is None:
            print("Milvus dependencies are not available. Did you install langchain-milvus and pymilvus?")
            vector_db = None
            return
        try:
            connection_args: dict = {"uri": MILVUS_URI}
            if MILVUS_TOKEN:
                connection_args["token"] = MILVUS_TOKEN
            if MILVUS_DB_NAME:
                connection_args["db_name"] = MILVUS_DB_NAME
            vector_db = MilvusVectorStore(
                embedding_function=embeddings,
                connection_args=connection_args,
                collection_name=MILVUS_COLLECTION,
            )
            print(f"Connected to Milvus collection '{MILVUS_COLLECTION}' at '{MILVUS_URI}'.")
        except Exception as e:  # noqa: BLE001
            print(f"Failed connecting to Milvus: {e}")
            vector_db = None
        return

    if FAISS is None or _faiss is None:
        print("FAISS dependencies not available. Install faiss-cpu and langchain-community.")
        vector_db = None
        return

    if (VECTOR_DIR / "index.faiss").exists() and (VECTOR_DIR / "index.pkl").exists():
        try:
            vector_db = FAISS.load_local(
                folder_path=str(VECTOR_DIR),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"FAISS VectorStore loaded from {VECTOR_DIR}")
        except Exception as e:  # noqa: BLE001
            print(f"Failed loading FAISS store from disk: {e}")
            vector_db = None
    else:
        try:
            dim = len(embeddings.embed_query("hello world"))
            index = _faiss.IndexFlatL2(dim)
            vector_db = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            print("Initialized empty FAISS vector store (no local files present).")
        except Exception as e:  
            print(f"Failed initializing empty FAISS index: {e}")
            vector_db = None


def load_paper_vector_store() -> None:
    """Initialize or connect to the dedicated vector store for papers."""
    global paper_vector_db

    if PAPER_VECTOR_DB_PROVIDER == "pinecone":
        if Pinecone is None or PineconeVectorStore is None:
            print("Pinecone dependencies are not available for paper vector store.")
            paper_vector_db = None
            return
        try:
            _pc = Pinecone(api_key=PINECONE_API_KEY)
            paper_vector_db = PineconeVectorStore(
                index_name=PAPER_PINECONE_INDEX_NAME,
                embedding=embeddings,
            )
            print(f"Connected to paper Pinecone index '{PAPER_PINECONE_INDEX_NAME}'.")
        except Exception as e:  # noqa: BLE001
            print(f"Failed connecting paper vector store to Pinecone: {e}")
            paper_vector_db = None
        return

    if PAPER_VECTOR_DB_PROVIDER == "milvus":
        if MilvusVectorStore is None:
            print("Milvus dependencies are not available for paper vector store.")
            paper_vector_db = None
            return
        try:
            connection_args: dict = {"uri": MILVUS_URI}
            if MILVUS_TOKEN:
                connection_args["token"] = MILVUS_TOKEN
            if MILVUS_DB_NAME:
                connection_args["db_name"] = MILVUS_DB_NAME
            paper_vector_db = MilvusVectorStore(
                embedding_function=embeddings,
                connection_args=connection_args,
                collection_name=PAPER_MILVUS_COLLECTION,
            )
            print(f"Connected to paper Milvus collection '{PAPER_MILVUS_COLLECTION}' at '{MILVUS_URI}'.")
        except Exception as e:  # noqa: BLE001
            print(f"Failed connecting paper vector store to Milvus: {e}")
            paper_vector_db = None
        return

    if FAISS is None or _faiss is None:
        print("FAISS dependencies not available for paper vector store.")
        paper_vector_db = None
        return

    if (PAPER_VECTOR_DIR / "index.faiss").exists() and (PAPER_VECTOR_DIR / "index.pkl").exists():
        try:
            paper_vector_db = FAISS.load_local(
                folder_path=str(PAPER_VECTOR_DIR),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"Paper FAISS VectorStore loaded from {PAPER_VECTOR_DIR}")
        except Exception as e:  # noqa: BLE001
            print(f"Failed loading paper FAISS store from disk: {e}")
            paper_vector_db = None
    else:
        try:
            dim = len(embeddings.embed_query("hello world"))
            index = _faiss.IndexFlatL2(dim)
            paper_vector_db = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            print("Initialized empty paper FAISS vector store (no local files present).")
        except Exception as e:
            print(f"Failed initializing empty paper FAISS index: {e}")
            paper_vector_db = None
