"""Connectors API (``/api/connectors``): apps whose tools chat can use.

The hub calls ``PUT /api/connectors/{id}`` for each app it installs that serves MCP, and
``DELETE`` when it removes one. Settings uses the same routes for connectors added by hand.
API keys are write-only: listings say whether one is set, never what it is.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app import connectors as conn
from app.security import require_api_key

router = APIRouter(prefix="/api/connectors", dependencies=[Depends(require_api_key)])


class ConnectorIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=60)
    url: str | None = Field(default=None, pattern=r"^https?://", max_length=500)
    # None: leave as it is; "": remove it.
    api_key: str | None = Field(default=None, max_length=500)
    enabled: bool | None = None
    source: str | None = Field(default=None, pattern=r"^(user|hub)$")


def _public(request: Request, c: dict) -> dict:
    status = request.app.state.connectors.status.get(c["id"])
    out = {k: v for k, v in c.items() if k != "api_key"} | {"has_key": bool(c.get("api_key"))}
    if status:
        out |= {
            "server": status.name,
            "error": status.error,
            "tools": [
                {
                    "name": t["name"],
                    "description": t.get("description") or t.get("title") or "",
                    "changes": conn.Connectors.wrap(request.app.state.connectors, c, t).mutating,
                }
                for t in status.tools
            ],
        }
    return out


def _get(request: Request, connector_id: str) -> dict:
    found = request.app.state.db.get_connector(connector_id)
    if found is None:
        raise HTTPException(404, "No such connector")
    return found


@router.get("")
async def list_connectors(request: Request) -> list[dict]:
    """Connectors, with what each offered the last time it was asked."""
    found = request.app.state.db.list_connectors()
    for c in found:
        if c["enabled"]:
            await request.app.state.connectors.refresh(c)
    return [_public(request, c) for c in found]


@router.put("/{connector_id}")
async def put_connector(request: Request, connector_id: str, body: ConnectorIn) -> dict:
    """Add a connector or change one. A new connector needs a URL."""
    if not conn.ID.match(connector_id):
        raise HTTPException(400, "Use lowercase letters, digits and dashes (up to 30) for the id")
    db = request.app.state.db
    fields = body.model_dump(exclude_none=True)
    if fields.get("api_key") == "":
        fields["api_key"] = None
    if db.get_connector(connector_id) is None and not fields.get("url"):
        raise HTTPException(400, "A new connector needs its MCP URL")
    saved = db.put_connector(connector_id, **fields)
    status = await request.app.state.connectors.refresh(saved, force=True)
    if not body.name and status.name and db.get_connector(connector_id)["name"] == connector_id:
        db.put_connector(connector_id, name=status.name[:60])
    return _public(request, _get(request, connector_id))


@router.post("/{connector_id}/test")
async def test_connector(request: Request, connector_id: str) -> dict:
    """Ask the app for its tools now."""
    await request.app.state.connectors.refresh(_get(request, connector_id), force=True)
    return _public(request, _get(request, connector_id))


@router.delete("/{connector_id}", status_code=204)
async def delete_connector(request: Request, connector_id: str) -> Response:
    if not request.app.state.db.delete_connector(connector_id):
        raise HTTPException(404, "No such connector")
    request.app.state.connectors.forget(connector_id)
    return Response(status_code=204)
