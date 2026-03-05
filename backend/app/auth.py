from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from app.models import User
from app.models import Session as UserSession
from app.models import Message
from app.schemas import UserCreate, UserOut, LoginRequest, Token, UserListResponse, UserMeta
from app.security import (
    get_db,
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_current_user,
    upload_allowed,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from app.user_roles import (
    allowed_roles,
    level_code_to_role,
    normalize_role_name,
    role_to_level_code,
)

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(
    payload: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if (current_user.Level or 0) != 1:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only level 1 users can create users")
    existing = db.query(User).filter(User.Username == payload.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    role = normalize_role_name(payload.role or "")
    if role is None and payload.level is not None:
        role = level_code_to_role(payload.level)
    if role is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid role. Allowed roles: {', '.join(allowed_roles())}",
        )
    user = User(
        Username=payload.username,
        HashedPassword=get_password_hash(payload.password),
        Level=role_to_level_code(role),
    )
    session = UserSession(session_id=payload.username)
    db.add(user)
    db.add(session)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.Id, username=user.Username, role=level_code_to_role(user.Level), level=user.Level)

@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        {"sub": user.Username, "lvl": user.Level, "role": level_code_to_role(user.Level)},
        expires_delta=access_token_expires,
    )
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserOut, status_code=status.HTTP_200_OK)
def me(current_user: User = Depends(get_current_user)):
    return UserOut(
        id=current_user.Id,
        username=current_user.Username,
        role=level_code_to_role(current_user.Level),
        level=current_user.Level,
        can_upload=upload_allowed(current_user),
    )

@router.get("/users", response_model=UserListResponse, status_code=status.HTTP_200_OK)
def list_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if (current_user.Level or 0) != 1:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only level 1 users can view users")
    rows = db.query(User).order_by(User.CreatedAt.desc()).all()
    items = [
        UserMeta(
            id=u.Id,
            username=u.Username,
            role=level_code_to_role(u.Level),
            level=u.Level,
            created_at=u.CreatedAt,
        )
        for u in rows
    ]
    return UserListResponse(items=items)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if (current_user.Level or 0) != 1:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only level 1 users can delete users")

    user = db.query(User).filter(User.Id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.Id == current_user.Id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete your own user account")

    sessions = db.query(UserSession).filter(UserSession.session_id == user.Username).all()
    if sessions:
        session_ids = [s.id for s in sessions]
        db.query(Message).filter(Message.session_id.in_(session_ids)).delete(synchronize_session=False)
        db.query(UserSession).filter(UserSession.id.in_(session_ids)).delete(synchronize_session=False)

    db.delete(user)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
