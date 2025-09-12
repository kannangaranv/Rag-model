from typing import Optional, Callable
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

from typing import Optional, Callable, List
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import asc

def load_session_history(
    session_id: str,
    *,
    # NEW: optional seed messages you want to add first
    seed_system: Optional[str] = None,
    seed_user: Optional[str] = None,
    seed_ai: Optional[str] = None,
    max_messages: Optional[int] = 30,
    max_tokens: Optional[int] = None,
    token_counter: Optional[Callable] = None,
) -> BaseChatMessageHistory:
    db = next(get_db())
    # 1) Always start with a fresh history
    history = ChatMessageHistory()

    # 2) Optionally add initial/seed messages BEFORE loading DB messages
    if seed_system:
        history.add_message(SystemMessage(seed_system))
    if seed_user:
        history.add_user_message(seed_user)
    if seed_ai:
        history.add_ai_message(seed_ai)

    try:
        sess = db.query(Session).filter(Session.session_id == session_id).first()
        if not sess:
            # No DB session → return just the seeded history (if any)
            return history

        order_col = getattr(Message, "created_at", None) or Message.id
        rows: List[Message] = (
            db.query(Message)
              .filter(Message.session_id == sess.id)
              .order_by(asc(order_col))
              .all()
        )

        # 3) Convert DB rows to LangChain messages
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
                # Unknown → treat as system to be safe
                db_msgs.append(SystemMessage(m.content))

        # Merge: seeds first, then DB
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
                token_counter=len,  # counts messages
                max_tokens=max_messages,
                include_system=True,
                start_on="human",
                end_on=("human", "tool"),
                allow_partial=False,
            )
        else:
            trimmed = merged

        # Return a new history built from the trimmed list
        return ChatMessageHistory(messages=trimmed)

    except SQLAlchemyError:
        # On error, at least return whatever seeds were added
        return history
    finally:
        db.close()
