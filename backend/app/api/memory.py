"""Long-term memory API (``/api/memory``): view, search, edit, add and forget facts."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app import memory
from app.security import require_api_key

router = APIRouter(prefix="/api/memory", dependencies=[Depends(require_api_key)])


class FactBody(BaseModel):
    text: str = Field(min_length=3, max_length=300)
    category: str = "general"


class FactUpdate(BaseModel):
    text: str | None = Field(default=None, min_length=3, max_length=300)
    category: str | None = None


class MemorySettings(BaseModel):
    enabled: bool | None = None
    auto: bool | None = None


def public(fact: dict) -> dict:
    return {k: v for k, v in fact.items() if k not in ("embedding", "embedding_model", "user_id")}


@router.get("")
async def list_facts(request: Request, q: str | None = None) -> dict:
    facts = await memory.search(request.app, q) if q else request.app.state.db.list_memories()
    us = request.app.state.user_settings
    return {
        "facts": [public(f) for f in facts],
        "enabled": us.memory_enabled,
        "auto": us.memory_auto,
        "embedding_model": await memory.embedding_model(request.app),
        "categories": list(memory.CATEGORIES),
    }


@router.post("", status_code=201)
async def add_fact(request: Request, body: FactBody) -> dict:
    text = memory.clean_fact(body.text)
    if not text:
        raise HTTPException(400, "That looks like sensitive information; it won't be stored.")
    category = body.category if body.category in memory.CATEGORIES else "general"
    return public(request.app.state.db.add_memory(text, category))


@router.patch("/{fact_id}")
async def update_fact(request: Request, fact_id: str, body: FactUpdate) -> dict:
    fields: dict = {}
    if body.text is not None:
        text = memory.clean_fact(body.text)
        if not text:
            raise HTTPException(400, "That looks like sensitive information; it won't be stored.")
        fields.update(text=text, embedding=None, embedding_model=None)  # re-embedded on next use
    if body.category is not None:
        fields["category"] = body.category if body.category in memory.CATEGORIES else "general"
    fact = request.app.state.db.update_memory(fact_id, **fields)
    if fact is None:
        raise HTTPException(404, "Fact not found")
    return public(fact)


@router.delete("/{fact_id}", status_code=204)
async def delete_fact(request: Request, fact_id: str) -> None:
    if not request.app.state.db.delete_memory(fact_id):
        raise HTTPException(404, "Fact not found")


@router.delete("")
async def forget_everything(request: Request, confirm: bool = False) -> dict:
    if not confirm:
        raise HTTPException(400, "Pass confirm=true to delete every fact.")
    return {"deleted": request.app.state.db.delete_all_memories()}


@router.put("/settings")
async def memory_settings(request: Request, body: MemorySettings) -> dict:
    us = request.app.state.user_settings
    us.set_memory(body.enabled, body.auto)
    return {"enabled": us.memory_enabled, "auto": us.memory_auto}
