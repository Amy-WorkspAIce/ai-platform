import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import catalog, delegation
from app.config import ProviderConfig, Settings
from app.engines import ENGINES
from app.main import create_app

AUTH = {"Authorization": "Bearer k"}
PNG = base64.b64encode(b"\x89PNG fake").decode()


def install(app, engine: str, tag: str, variant: dict | None = None):
    engines = app.state.engines
    engines.root.joinpath(engine).mkdir(parents=True, exist_ok=True)
    engines._marker(engine).write_text(
        json.dumps({"version": ENGINES[engine].version, "gpu": engines.gpu})
    )
    d = engines.model_dir(engine, tag)
    d.mkdir(parents=True)
    (d / ".amy-model.json").write_text(json.dumps(variant or catalog.find_variant(tag)[1]))


class FakeEngines:
    def __init__(self, supports_image=True):
        self.calls = []
        self.supports_image = supports_image

    async def __call__(self, engine, path, payload, timeout=0):
        self.calls.append((path, payload))
        if path == "/generate":
            return httpx.Response(
                200,
                json={
                    "images": [PNG],
                    "seed": 11,
                    "edited": bool(payload.get("init_image")) and self.supports_image,
                },
            )
        if path == "/supports_image":
            return httpx.Response(200, json={"supports_image": self.supports_image})
        if path == "/park":
            return httpx.Response(200, json={"parked": False})
        if path == "/speak":
            return httpx.Response(200, content=b"RIFFwav", headers={"content-type": "audio/wav"})
        if path == "/generate_video":
            Path(payload["output"]).write_bytes(b"mp4")
            return httpx.Response(
                200, json={"seed": 3, "frames": payload["frames"], "fps": payload["fps"]}
            )
        return httpx.Response(404)


def sse(chunks):
    return (
        "".join(f"data: {json.dumps({'choices': [{'delta': c}]})}\n\n" for c in chunks)
        + "data: [DONE]\n\n"
    ).encode()


def tool_call(name, args):
    return [
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"c_{name}",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
        }
    ]


def scripted(turns, seen):
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, content=sse(turns[len(seen) - 1]), headers={"content-type": "text/event-stream"}
        )

    return handler


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = Settings(
        data_dir=tmp_path,
        api_key="k",
        ollama_url="",
        static_dir=tmp_path,
        gpu_vram_gb=11,
        providers=[ProviderConfig(name="fake", base_url="http://fake/v1")],
    )
    with TestClient(create_app(settings)) as c:
        c.fake = FakeEngines()
        monkeypatch.setattr(c.app.state.engines, "call", c.fake)
        # Pretend the image worker is running (so results get "parked"); already exited for shutdown.
        running = SimpleNamespace(proc=SimpleNamespace(returncode=0))
        monkeypatch.setattr(c.app.state.engines, "workers", {"image": running})
        yield c


def chat(client, turns, message="hi", conversation_id=None):
    seen = []
    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(scripted(turns, seen)))
    body = {"model": "fake/m", "message": message, "conversation_id": conversation_id}
    with client.stream("POST", "/api/agent/chat", headers=AUTH, json=body) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    return events, seen


# ---- routing ----


def test_best_installed_pinned_and_requested(client):
    app = client.app
    assert delegation.choose(app, "image") is None
    install(app, "image", "sd15")
    install(app, "image", "flux2-klein-4b")
    ranked = [m["id"] for m in delegation.installed_for(app, "image")]
    assert ranked[0] == "image/flux2-klein-4b"  # more capable, still fits 11 GB
    assert delegation.choose(app, "image")["id"] == "image/flux2-klein-4b"
    assert delegation.choose(app, "image", requested="Stable Diffusion 1.5")["id"] == "image/sd15"
    r = client.put(
        "/api/settings/models", headers=AUTH, json={"capability": "image", "model": "image/sd15"}
    )
    assert r.json()["image"]["pinned"] == "image/sd15"
    assert delegation.choose(app, "image")["id"] == "image/sd15"
    client.put("/api/settings/models", headers=AUTH, json={"capability": "image", "model": None})
    assert delegation.choose(app, "image")["id"] == "image/flux2-klein-4b"
    bad = client.put(
        "/api/settings/models", headers=AUTH, json={"capability": "image", "model": "image/nope"}
    )
    assert bad.status_code == 400


def test_tools_offered_only_for_installed_media(client):
    _, seen = chat(client, [[{"content": "hello"}]])
    names = {t["function"]["name"] for t in seen[0].get("tools", [])}
    assert names == {"web_search", "fetch_url", "get_weather"}
    install(client.app, "image", "sd15")
    install(client.app, "audio", "kokoro-82m")
    _, seen = chat(client, [[{"content": "hello"}]])
    names = {t["function"]["name"] for t in seen[0]["tools"]}
    assert {"generate_image", "edit_image", "speak"} <= names and "generate_video" not in names
    assert "generate_image / edit_image" in seen[0]["messages"][0]["content"]


def test_web_access_setting(client):
    client.put("/api/settings/web", headers=AUTH, json={"enabled": False})
    _, seen = chat(client, [[{"content": "hello"}]])
    assert "tools" not in seen[0]
    assert "turned off in Settings" in seen[0]["messages"][0]["content"]


# ---- delegated work ----


def test_generate_then_edit_image(client):
    install(client.app, "image", "flux2-klein-4b")
    events, seen = chat(
        client,
        [
            tool_call("generate_image", {"prompt": "a cabin", "aspect": "wide"}),
            [{"content": "Here it is."}],
        ],
    )
    result = next(e for e in events if e["type"] == "tool_result")
    image_id = result["attachments"][0]["id"]
    assert result["ok"] and result["attachments"][0]["type"] == "image"
    gen = [p for path, p in client.fake.calls if path == "/generate"][0]
    assert gen["width"] > gen["height"]  # wide
    assert image_id in seen[1]["messages"][-1]["content"]  # the model learns the id

    conv = events[0]["conversation_id"]
    events, _ = chat(
        client,
        [
            tool_call("edit_image", {"image_id": image_id, "instruction": "make it nighttime"}),
            [{"content": "Done."}],
        ],
        conversation_id=conv,
    )
    edit = [p for path, p in client.fake.calls if path == "/generate"][-1]
    assert edit["prompt"] == "make it nighttime" and edit["init_image"].endswith(f"{image_id}.png")
    new = next(e for e in events if e["type"] == "tool_result")["attachments"][0]["id"]
    assert new != image_id
    saved = client.get(f"/api/conversations/{conv}", headers=AUTH).json()["messages"]
    assert any((m.get("meta") or {}).get("attachments") for m in saved)


def test_edit_falls_back_to_regenerating(client):
    client.fake.supports_image = False
    install(client.app, "image", "sdxl-base-1.0")
    events, _ = chat(
        client, [tool_call("generate_image", {"prompt": "a cabin"}), [{"content": "ok"}]]
    )
    image_id = next(e for e in events if e["type"] == "tool_result")["attachments"][0]["id"]
    events, _ = chat(
        client,
        [
            tool_call("edit_image", {"image_id": image_id, "instruction": "at night"}),
            [{"content": "ok"}],
        ],
    )
    regen = [p for path, p in client.fake.calls if path == "/generate"][-1]
    assert (
        regen["prompt"] == "a cabin. at night" and regen["seed"] == 11 and not regen["init_image"]
    )
    assert (
        "can't edit images directly"
        in next(e for e in events if e["type"] == "tool_result")["output"]
    )


def test_generate_video_and_speak(client):
    install(client.app, "image", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    install(client.app, "audio", "kokoro-82m")
    events, _ = chat(
        client,
        [
            tool_call("generate_video", {"prompt": "waves", "seconds": 3}),
            [{"content": "On its way."}],
        ],
    )
    att = next(e for e in events if e["type"] == "tool_result")["attachments"][0]
    assert att["type"] == "video" and att["job_id"] in client.app.state.video_jobs
    events, _ = chat(client, [tool_call("speak", {"text": "hello"}), [{"content": "ok"}]])
    audio_id = next(e for e in events if e["type"] == "tool_result")["attachments"][0]["id"]
    assert client.get(f"/api/audio/{audio_id}", headers=AUTH).content == b"RIFFwav"
    assert client.get("/api/audio", headers=AUTH).json()[0]["text"] == "hello"


def test_missing_model_is_explained_to_the_model(client):
    install(client.app, "image", "sd15")
    events, seen = chat(client, [tool_call("generate_video", {"prompt": "x"}), [{"content": "ok"}]])
    assert (
        "Unknown tool" in seen[1]["messages"][-1]["content"]
    )  # no video model -> tool not offered


def test_model_without_tools_falls_back_to_plain_chat(client):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "tools" in body:
            return httpx.Response(400, json={"error": {"message": "m does not support tools"}})
        return httpx.Response(
            200,
            content=sse([{"content": "plain answer"}]),
            headers={"content-type": "text/event-stream"},
        )

    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with client.stream(
        "POST", "/api/agent/chat", headers=AUTH, json={"model": "fake/m", "message": "hi"}
    ) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    assert any(e["type"] == "notice" for e in events)
    assert "".join(e["delta"] for e in events if e["type"] == "text") == "plain answer"
    assert "tools" in calls[0] and "tools" not in calls[1]


def test_upload_image_for_editing(client):
    r = client.post(
        "/api/images/upload",
        headers=AUTH,
        files={"file": ("photo.jpg", b"\xff\xd8jpeg", "image/jpeg")},
    )
    image_id = r.json()["id"]
    assert client.get(f"/api/images/{image_id}", headers=AUTH).content == b"\xff\xd8jpeg"
    bad = client.post(
        "/api/images/upload", headers=AUTH, files={"file": ("x.txt", b"hi", "text/plain")}
    )
    assert bad.status_code == 400
