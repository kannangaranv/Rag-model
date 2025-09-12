from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models import User
from app.models import Session as UserSession
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

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.Username == payload.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    if payload.level < 2 or payload.level > 6:
        raise HTTPException(status_code=422, detail="Level must be between 2 and 6")
    user = User(Username=payload.username, HashedPassword=get_password_hash(payload.password), Level=payload.level)
    session = UserSession(session_id=payload.username)
    db.add(user)
    db.add(session)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.Id, username=user.Username, level=user.Level)

@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token({"sub": user.Username, "lvl": user.Level}, expires_delta=access_token_expires)
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserOut, status_code=status.HTTP_200_OK)
def me(current_user: User = Depends(get_current_user)):
    return UserOut(
        id=current_user.Id,
        username=current_user.Username,
        level=current_user.Level,
        can_upload=upload_allowed(current_user),
    )

@router.get("/users", response_model=UserListResponse, status_code=status.HTTP_200_OK)
def list_users(db: Session = Depends(get_db)):
    rows = db.query(User).order_by(User.CreatedAt.desc()).all()
    items = [UserMeta(id=u.Id, username=u.Username, level=u.Level, created_at=u.CreatedAt) for u in rows]
    return UserListResponse(items=items)


