"""Saved chat conversations (``/api/conversations``)."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.db import Database
from app.security import require_api_key

router = APIRouter(prefix="/api/conversations", dependencies=[Depends(require_api_key)])


def get_db(request: Request) -> Database:
    return request.app.state.db


class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", max_length=200)
    model: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    model: str | None = None


class MessageCreate(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    reasoning: str | None = None
    model: str | None = None
    meta: dict | None = None


@router.get("")
def list_conversations(db: Database = Depends(get_db)) -> list[dict]:
    return db.list_conversations()


@router.post("", status_code=201)
def create_conversation(body: ConversationCreate, db: Database = Depends(get_db)) -> dict:
    return db.create_conversation(body.title.strip() or "New chat", body.model)


@router.get("/{conv_id}")
def get_conversation(conv_id: str, db: Database = Depends(get_db)) -> dict:
    conv = db.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return conv


@router.patch("/{conv_id}")
def update_conversation(
    conv_id: str, body: ConversationUpdate, db: Database = Depends(get_db)
) -> dict:
    conv = db.update_conversation(conv_id, title=body.title, model=body.model)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return conv


@router.delete("/{conv_id}", status_code=204)
def delete_conversation(conv_id: str, db: Database = Depends(get_db)) -> None:
    if not db.delete_conversation(conv_id):
        raise HTTPException(404, "Conversation not found")


@router.post("/{conv_id}/messages", status_code=201)
def add_message(conv_id: str, body: MessageCreate, db: Database = Depends(get_db)) -> dict:
    msg = db.add_message(conv_id, body.role, body.content, body.reasoning, body.model, body.meta)
    if msg is None:
        raise HTTPException(404, "Conversation not found")
    return msg
