from fastapi import APIRouter, HTTPException, Depends
from models.chat import ChatRequest
from database import get_db
from routers.auth import get_current_user
from services.rag import get_rag_response
from datetime import datetime
from bson import ObjectId

router = APIRouter()

FREE_QUERIES_PER_DAY = 20


@router.post("/")
async def chat(request: ChatRequest, current_user=Depends(get_current_user)):
    db = get_db()
    user_id = str(current_user["_id"])
    plan = current_user.get("plan", "free")

    # Enforce daily query limit on free plan
    if plan == "free":
        today = datetime.utcnow().date().isoformat()
        if current_user.get("queries_date") != today:
            await db.users.update_one(
                {"_id": current_user["_id"]},
                {"$set": {"queries_today": 0, "queries_date": today}},
            )
            queries_today = 0
        else:
            queries_today = current_user.get("queries_today", 0)

        if queries_today >= FREE_QUERIES_PER_DAY:
            raise HTTPException(
                status_code=429,
                detail=f"Daily query limit ({FREE_QUERIES_PER_DAY}) reached. Upgrade to Pro for unlimited queries!",
            )

    # Fetch very recent chat history (Reduced to 3 to save tokens)
    history_cursor = db.chats.find({"user_id": user_id}).sort("created_at", -1).limit(3)
    history = await history_cursor.to_list(length=3)
    history.reverse()

    # Run RAG pipeline
    try:
        result = await get_rag_response(
            user_id=user_id,
            message=request.message,
            document_ids=request.document_ids,
            chat_history=history,
        )
    except Exception as e:
        import traceback
        with open("C:/knowbase/chat_error.log", "w") as f:
            f.write(traceback.format_exc())
        raise HTTPException(status_code=500, detail="RAG Error")

    # Persist chat to DB
    chat_doc = {
        "user_id": user_id,
        "message": request.message,
        "response": result["answer"],
        "sources": result["sources"],
        "created_at": datetime.utcnow(),
    }
    try:
        res = await db.chats.insert_one(chat_doc)
    except Exception as e:
        import traceback
        with open("C:/knowbase/chat_error.log", "a") as f:
            f.write("\n\nDB Insert Error:\n")
            f.write(traceback.format_exc())
        raise HTTPException(status_code=500, detail="DB Error")

    # Increment daily query counter
    await db.users.update_one(
        {"_id": current_user["_id"]},
        {"$inc": {"queries_today": 1}},
    )

    return {
        "id": str(res.inserted_id),
        "message": request.message,
        "response": result["answer"],
        "sources": result["sources"],
        "created_at": chat_doc["created_at"],
    }


@router.get("/history")
async def get_history(
    skip: int = 0, limit: int = 50, current_user=Depends(get_current_user)
):
    db = get_db()
    user_id = str(current_user["_id"])

    cursor = (
        db.chats.find({"user_id": user_id})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    chats = await cursor.to_list(length=limit)

    return [
        {
            "id": str(c["_id"]),
            "message": c["message"],
            "response": c["response"],
            "sources": c.get("sources", []),
            "created_at": c["created_at"],
        }
        for c in chats
    ]


@router.delete("/history")
async def clear_history(current_user=Depends(get_current_user)):
    db = get_db()
    user_id = str(current_user["_id"])
    result = await db.chats.delete_many({"user_id": user_id})
    return {"message": f"Cleared {result.deleted_count} chat messages"}
