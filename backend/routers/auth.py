from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from models.user import UserCreate, UserLogin, UserResponse, Token
from database import get_db
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from bson import ObjectId
import os

router = APIRouter()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("JWT_SECRET", "change-this-secret-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    db = get_db()
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.post("/signup", response_model=Token)
async def signup(user_data: UserCreate):
    db = get_db()

    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = pwd_context.hash(user_data.password[:72])
    today = datetime.utcnow().date().isoformat()

    user_doc = {
        "name": user_data.name,
        "email": user_data.email,
        "password": hashed_password,
        "plan": "free",
        "queries_today": 0,
        "queries_date": today,
        "total_docs": 0,
        "created_at": datetime.utcnow(),
    }

    result = await db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)
    token = create_access_token({"sub": user_id})

    return Token(
        access_token=token,
        token_type="bearer",
        user=UserResponse(
            id=user_id,
            name=user_data.name,
            email=user_data.email,
            plan="free",
            created_at=user_doc["created_at"],
            queries_today=0,
            total_docs=0,
        ),
    )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    db = get_db()

    user = await db.users.find_one({"email": credentials.email})
    if not user or not pwd_context.verify(credentials.password[:72], user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Reset daily query count if new day
    today = datetime.utcnow().date().isoformat()
    if user.get("queries_date") != today:
        await db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"queries_today": 0, "queries_date": today}},
        )
        user["queries_today"] = 0

    token = create_access_token({"sub": str(user["_id"])})

    return Token(
        access_token=token,
        token_type="bearer",
        user=UserResponse(
            id=str(user["_id"]),
            name=user["name"],
            email=user["email"],
            plan=user.get("plan", "free"),
            created_at=user["created_at"],
            queries_today=user.get("queries_today", 0),
            total_docs=user.get("total_docs", 0),
        ),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_user)):
    return UserResponse(
        id=str(current_user["_id"]),
        name=current_user["name"],
        email=current_user["email"],
        plan=current_user.get("plan", "free"),
        created_at=current_user["created_at"],
        queries_today=current_user.get("queries_today", 0),
        total_docs=current_user.get("total_docs", 0),
    )


class UserUpdate(BaseModel):
    name: str = Field(None, min_length=2, max_length=50)


@router.patch("/me", response_model=UserResponse)
async def update_me(update: UserUpdate, current_user=Depends(get_current_user)):
    db = get_db()
    updates = {}
    if update.name:
        updates["name"] = update.name.strip()
    if updates:
        await db.users.update_one({"_id": current_user["_id"]}, {"$set": updates})
    updated = await db.users.find_one({"_id": current_user["_id"]})
    return UserResponse(
        id=str(updated["_id"]),
        name=updated["name"],
        email=updated["email"],
        plan=updated.get("plan", "free"),
        created_at=updated["created_at"],
        queries_today=updated.get("queries_today", 0),
        total_docs=updated.get("total_docs", 0),
    )


class PlanUpdate(BaseModel):
    plan: str = Field(..., pattern="^(free|pro)$")


@router.patch("/me/plan", response_model=UserResponse)
async def update_plan(update: PlanUpdate, current_user=Depends(get_current_user)):
    """Demo endpoint — toggles Free/Pro plan for presentation purposes."""
    db = get_db()
    await db.users.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"plan": update.plan}},
    )
    updated = await db.users.find_one({"_id": current_user["_id"]})
    return UserResponse(
        id=str(updated["_id"]),
        name=updated["name"],
        email=updated["email"],
        plan=updated.get("plan", "free"),
        created_at=updated["created_at"],
        queries_today=updated.get("queries_today", 0),
        total_docs=updated.get("total_docs", 0),
    )

