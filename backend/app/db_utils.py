from typing import Optional, Callable
from datetime import datetime, timedelta

from sqlalchemy import asc
from sqlalchemy.exc import SQLAlchemyError

from app.config import SessionLocal
from app.models import Session, Message

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately



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
    max_messages: Optional[int] = 30,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable] = None,  
) -> BaseChatMessageHistory:
    db = next(get_db())
    try:
        sess = db.query(Session).filter(Session.session_id == session_id).first()
        if not sess:
            return ChatMessageHistory()

        order_col = getattr(Message, "created_at", None) or Message.id
        rows = (
            db.query(Message)
              .filter(Message.session_id == sess.id)
              .order_by(asc(order_col))
              .all()
        )

        print(f"Loaded {len(rows)} messages for session_id={session_id}")

        msgs = []
        for m in rows:
            role = (m.role or "").lower()
            if role in ("user", "human"):
                msgs.append(HumanMessage(m.content))
            elif role in ("assistant", "ai"):
                msgs.append(AIMessage(m.content))
            elif role == "system":
                msgs.append(SystemMessage(m.content))
            else:
                msgs.append(SystemMessage(m.content))

        if max_tokens is not None:
            tc = token_counter or count_tokens_approximately
            trimmed = trim_messages(
                msgs,
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
                msgs,
                strategy="last",
                token_counter=len,         
                max_tokens=max_messages,
                include_system=True,
                start_on="human",
                end_on=("human", "tool"),
                allow_partial=False,
            )
        else:
            trimmed = msgs
        return ChatMessageHistory(messages=trimmed)
    except SQLAlchemyError:
        return ChatMessageHistory()
    finally:
        db.close()
