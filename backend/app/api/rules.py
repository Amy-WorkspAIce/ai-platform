"""Rules API (``/api/rules``): standing instructions, everywhere or for one folder or workspace."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from app import rules as rulesets
from app.api.agent import get_workspace
from app.security import require_api_key

router = APIRouter(prefix="/api/rules", dependencies=[Depends(require_api_key)])

Applies = Literal["always", "chat", "agent"]


class NewRule(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=5000)
    scope: str = "global"
    applies: Applies = "always"


class RulePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    text: str | None = Field(default=None, min_length=1, max_length=5000)
    applies: Applies | None = None
    enabled: bool | None = None


class Exclusion(BaseModel):
    scope: str
    excluded: bool


def check_scope(request: Request, scope: str) -> str:
    """The scope, if it names something that exists: global, a folder or a workspace."""
    kind, _, name = scope.partition(":")
    if scope == "global":
        return scope
    if kind == "folder" and name in {f["id"] for f in request.app.state.db.list_folders()}:
        return scope
    if kind == "workspace" and name:
        get_workspace(request, name)  # 400/404 if it isn't one
        return scope
    raise HTTPException(404, f"No such folder or workspace: '{scope}'")


@router.get("")
async def list_rules(request: Request, scope: str = Query("global")) -> dict:
    """A scope's own rules, the global rules it inherits (and which it turned off), and for a
    workspace, the rule files already in it."""
    scope = check_scope(request, scope)
    db = request.app.state.db
    out = {"scope": scope, "rules": db.list_rules([scope]), "inherited": [], "project": []}
    if scope != "global":
        excluded = db.rule_exclusions([scope])
        out["inherited"] = [
            r | {"excluded": r["id"] in excluded} for r in db.list_rules(["global"])
        ]
    if scope.startswith("workspace:"):
        ws = get_workspace(request, scope.split(":", 1)[1])
        out["project"] = rulesets.project_rules(ws.root)
    return out


@router.post("", status_code=201)
async def add_rule(request: Request, body: NewRule) -> dict:
    scope = check_scope(request, body.scope)
    return request.app.state.db.add_rule(body.name.strip(), body.text.strip(), scope, body.applies)


@router.patch("/{rule_id}")
async def update_rule(request: Request, rule_id: str, body: RulePatch) -> dict:
    fields = body.model_dump(exclude_none=True)
    for key in ("name", "text"):
        if key in fields:
            fields[key] = fields[key].strip()
    rule = request.app.state.db.update_rule(rule_id, **fields)
    if rule is None:
        raise HTTPException(404, "Rule not found")
    return rule


@router.delete("/{rule_id}", status_code=204)
async def delete_rule(request: Request, rule_id: str) -> Response:
    if not request.app.state.db.delete_rule(rule_id):
        raise HTTPException(404, "Rule not found")
    return Response(status_code=204)


@router.put("/{rule_id}/excluded")
async def set_excluded(request: Request, rule_id: str, body: Exclusion) -> dict:
    """Turn a global rule off (or back on) for one folder or workspace."""
    db = request.app.state.db
    rule = db.get_rule(rule_id)
    if rule is None:
        raise HTTPException(404, "Rule not found")
    if rule["scope"] != "global":
        raise HTTPException(400, "Only rules that apply everywhere can be turned off for one place")
    scope = check_scope(request, body.scope)
    if scope == "global":
        raise HTTPException(400, "Turn the rule off instead")
    db.set_rule_excluded(scope, rule_id, body.excluded)
    return {"rule_id": rule_id, "scope": scope, "excluded": body.excluded}
