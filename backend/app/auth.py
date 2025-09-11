from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
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
    require_levels,
    upload_allowed,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db), _: User = Depends(require_levels([1]))):
    existing = db.query(User).filter(User.Username == payload.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    # Admin can create only non-admin users: 2..6
    if payload.level < 2 or payload.level > 6:
        raise HTTPException(status_code=422, detail="Level must be between 2 and 6")
    user = User(Username=payload.username, HashedPassword=get_password_hash(payload.password), Level=payload.level)
    session = UserSession(session_id=payload.username)
    db.add(user)
    db.add(session)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.Id, username=user.Username, level=user.Level)


@router.post("/login", response_model=Token)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token({"sub": user.Username, "lvl": user.Level}, expires_delta=access_token_expires)
    return Token(access_token=token, token_type="bearer")


@router.post("/token", response_model=Token)
def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token({"sub": user.Username, "lvl": user.Level}, expires_delta=access_token_expires)
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return UserOut(
        id=current_user.Id,
        username=current_user.Username,
        level=current_user.Level,
        can_upload=upload_allowed(current_user),
    )


@router.get("/users", response_model=UserListResponse)
def list_users(_: User = Depends(require_levels([1])), db: Session = Depends(get_db)):
    rows = db.query(User).order_by(User.CreatedAt.desc()).all()
    items = [UserMeta(id=u.Id, username=u.Username, level=u.Level, created_at=u.CreatedAt) for u in rows]
    return UserListResponse(items=items)

@router.patch("/users/{user_id}/level", response_model=UserOut)
def update_user_level(user_id: int, new_level: int, db: Session = Depends(get_db), _: User = Depends(require_levels([1]))):
    user = db.query(User).filter(User.Id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if new_level < 2 or new_level > 6:
        raise HTTPException(status_code=422, detail="Level must be between 2 and 6")
    user.Level = int(new_level)
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.Id, username=user.Username, level=user.Level)
