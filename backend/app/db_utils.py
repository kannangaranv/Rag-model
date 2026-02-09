from typing import Optional, Callable
from sqlalchemy import asc, text
from sqlalchemy.exc import SQLAlchemyError
from app.config import SessionLocal
from app.models import Session, Message
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from typing import Optional, Callable, List
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import asc

# Dependency generator for database sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Save a message to the database
def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

def load_session_history(
    session_id: str,
    *,
    seed_system: Optional[str] = None,
    seed_user: Optional[str] = None,
    seed_ai: Optional[str] = None,
    max_messages: Optional[int] = 30,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable] = None,
) -> BaseChatMessageHistory:
    db = next(get_db())

    history = ChatMessageHistory()

    try:
        sess = db.query(Session).filter(Session.session_id == session_id).first()
        if not sess:
            return history

        order_col = getattr(Message, "created_at", None) or Message.id
        rows: List[Message] = (
            db.query(Message)
              .filter(Message.session_id == sess.id)
              .order_by(asc(order_col))
              .all()
        )

        db_msgs = []
        for m in rows:
            role = (m.role or "").lower()
            if role in ("user", "human"):
                db_msgs.append(HumanMessage(m.content))
            elif role in ("assistant", "ai"):
                db_msgs.append(AIMessage(m.content))
            elif role == "system":
                db_msgs.append(SystemMessage(m.content))
            else:
                db_msgs.append(SystemMessage(m.content))

        merged = history.messages + db_msgs

        # 4) Trim after merging (so seeds are considered too)
        if max_tokens is not None:
            tc = token_counter or count_tokens_approximately
            trimmed = trim_messages(
                merged,
                strategy="last",
                token_counter=tc,
                max_tokens=max_tokens,
                include_system=True,
                start_on="human",
                end_on=("human", "tool"),
                allow_partial=False,
            )
        elif max_messages is not None:
            trimmed = trim_messages(
                merged,
                strategy="last",
                token_counter=len,  
                max_tokens=max_messages,
                include_system=True,
                start_on="human",
                end_on=("human", "tool"),
                allow_partial=False,
            )
        else:
            trimmed = merged

        return ChatMessageHistory(messages=trimmed)

    except SQLAlchemyError:
        return history
    finally:
        db.close()

def upsert_knowledge_profile(doc_id: str, source_type: str, file_name: str | None, profile_text: str):
    with SessionLocal() as db:
        db.execute(
            text("""
                MERGE dbo.KnowledgeProfiles AS target
                USING (
                    SELECT
                        CONVERT(uniqueidentifier, :DocId) AS DocId,
                        :SourceType AS SourceType
                ) AS source
                ON target.DocId = source.DocId
                   AND target.SourceType = source.SourceType
                WHEN MATCHED THEN
                    UPDATE SET
                        FileName = :FileName,
                        ProfileText = :ProfileText,
                        UpdatedAt = SYSUTCDATETIME()
                WHEN NOT MATCHED THEN
                    INSERT (DocId, SourceType, FileName, ProfileText)
                    VALUES (CONVERT(uniqueidentifier, :DocId), :SourceType, :FileName, :ProfileText);
            """),
            {
                "DocId": doc_id,
                "SourceType": source_type,
                "FileName": file_name,
                "ProfileText": profile_text,
            }
        )
        db.commit()

def delete_knowledge_profile(doc_id: str, source_type: str):
    with SessionLocal() as db:
        db.execute(
            text("""
                DELETE FROM dbo.KnowledgeProfiles
                WHERE DocId = CONVERT(uniqueidentifier, :DocId)
                  AND SourceType = :SourceType
            """),
            {"DocId": doc_id, "SourceType": source_type},
        )
        db.commit()
