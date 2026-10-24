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
    folder_id: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    model: str | None = None
    folder_id: str | None = None  # "" = move out of its folder


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
    conv = db.create_conversation(body.title.strip() or "New chat", body.model)
    if body.folder_id:
        conv = db.update_conversation(conv["id"], folder_id=body.folder_id) or conv
    return conv


@router.get("/{conv_id}")
def get_conversation(conv_id: str, db: Database = Depends(get_db)) -> dict:
    conv = db.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return conv


@router.get("/{conv_id}/details")
def conversation_details(conv_id: str, db: Database = Depends(get_db)) -> dict:
    """The chat's Details panel: what memory did here, and the sub-agents it used."""
    if db.get_conversation(conv_id) is None:
        raise HTTPException(404, "Conversation not found")
    return {"memories": db.memory_log(conv_id), "subagents": db.list_subagents(conv_id)}


@router.patch("/{conv_id}")
def update_conversation(
    conv_id: str, body: ConversationUpdate, db: Database = Depends(get_db)
) -> dict:
    if body.folder_id and body.folder_id not in {f["id"] for f in db.list_folders()}:
        raise HTTPException(404, "Folder not found")
    conv = db.update_conversation(
        conv_id, title=body.title, model=body.model, folder_id=body.folder_id
    )
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


# ---- folders ----

folders = APIRouter(prefix="/api/folders", dependencies=[Depends(require_api_key)])


class FolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class FolderUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    collapsed: bool | None = None
    position: int | None = None


@folders.get("")
def list_folders(db: Database = Depends(get_db)) -> list[dict]:
    return db.list_folders()


@folders.post("", status_code=201)
def create_folder(body: FolderCreate, db: Database = Depends(get_db)) -> dict:
    return db.create_folder(body.name.strip())


@folders.patch("/{folder_id}")
def update_folder(folder_id: str, body: FolderUpdate, db: Database = Depends(get_db)) -> dict:
    folder = db.update_folder(
        folder_id, name=body.name, collapsed=body.collapsed, position=body.position
    )
    if folder is None:
        raise HTTPException(404, "Folder not found")
    return folder


@folders.delete("/{folder_id}", status_code=204)
def delete_folder(folder_id: str, db: Database = Depends(get_db)) -> None:
    """Delete a folder. Its chats are kept, just no longer in a folder."""
    if not db.delete_folder(folder_id):
        raise HTTPException(404, "Folder not found")
