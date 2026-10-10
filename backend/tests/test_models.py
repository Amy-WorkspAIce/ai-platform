import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import catalog
from app.config import Settings
from app.hardware import Gpu, Hardware, parse_nvidia_smi
from app.jobs import PullState
from app.main import create_app
from app.ollama_admin import apply_progress, valid_model_name

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}

RTX_2080_TI = Hardware(gpus=[Gpu("RTX 2080 Ti", 11.0, 10.0)], ram_total_gb=64, disk_free_gb=500)
CPU_ONLY = Hardware(ram_total_gb=16, disk_free_gb=500)


def test_parse_nvidia_smi():
    gpus = parse_nvidia_smi("NVIDIA GeForce RTX 2080 Ti, 11264, 10240\ngarbage line\n")
    assert gpus == [Gpu("NVIDIA GeForce RTX 2080 Ti", 11.0, 10.0)]


@pytest.mark.parametrize(
    ("size", "hw", "fit"),
    [
        (5.2, RTX_2080_TI, "gpu"),
        (8.2, RTX_2080_TI, "gpu"),
        (9.3, RTX_2080_TI, "partial"),
        (13.8, RTX_2080_TI, "partial"),
        (65.4, RTX_2080_TI, "too_large"),
        (2.5, CPU_ONLY, "cpu"),
        (20, CPU_ONLY, "too_large"),
        (2.5, Hardware(), "unknown"),
    ],
)
def test_assess_fit(size, hw, fit):
    assert catalog.assess_fit(size, hw) == fit


def test_recommendations_pick_largest_gpu_fit():
    recs = catalog.build(RTX_2080_TI, set())["recommendations"]
    assert recs["chat"][0] == "qwen3:8b"  # 14b needs ~11.7 GB with overhead
    assert recs["image"][0] == "flux2-klein-4b"
    assert recs["speech"] == ["kokoro-82m"]
    for tags in recs.values():
        assert 1 <= len(tags) <= 3


def test_cpu_recommendations_stay_small():
    built = catalog.build(CPU_ONLY, set())
    sizes = {v["tag"]: v["size_gb"] for f in built["families"] for v in f["variants"]}
    for tags in built["recommendations"].values():
        assert all(sizes[t] <= catalog.CPU_RECOMMEND_MAX_GB for t in tags)


def test_variant_use_case_override():
    built = catalog.build(RTX_2080_TI, set())
    gemma = next(f for f in built["families"] if f["id"] == "gemma3")
    assert gemma["variants"][0]["use_cases"] == ["chat"]
    assert "gemma3:1b" not in built["recommendations"]["vision"]


def test_catalog_tags_unique():
    tags = [v["tag"] for f in catalog.load()["families"] for v in f["variants"]]
    assert len(tags) == len(set(tags))


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("qwen3:8b", True),
        ("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", True),
        ("../etc/passwd", False),
        ("a b", False),
        ("", False),
    ],
)
def test_valid_model_name(name, ok):
    assert valid_model_name(name) is ok


def test_pull_progress_aggregates_layers():
    s = PullState("m")
    apply_progress(s, {"status": "pulling manifest"})
    apply_progress(s, {"status": "pulling a", "digest": "a", "total": 100, "completed": 50})
    apply_progress(s, {"status": "pulling b", "digest": "b", "total": 300, "completed": 0})
    apply_progress(s, {"status": "pulling a", "digest": "a", "total": 100, "completed": 100})
    assert (s.completed, s.total, s.downloaded) == (100, 400, False)
    apply_progress(s, {"status": "success"})
    assert s.downloaded and not s.done  # done only after warm-up
    assert s.completed == 400 and s.error is None


def test_pull_error():
    s = PullState("m")
    apply_progress(s, {"error": "pull model manifest: file does not exist"})
    assert "does not exist" in s.error


WARMED: list[str] = []


def fake_ollama(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "size": 5_230_000_000}]})
    if request.url.path == "/api/pull":
        lines = [
            {"status": "pulling manifest"},
            {"status": "pulling x", "digest": "x", "total": 10, "completed": 10},
            {"status": "success"},
        ]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines))
    if request.url.path == "/api/generate":
        WARMED.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"done": True})
    if request.url.path == "/api/delete":
        return httpx.Response(200)
    return httpx.Response(404)


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        api_key=KEY,
        ollama_url="http://ollama:11434",
        static_dir=tmp_path,
        gpu_vram_gb=11,
    )
    with TestClient(create_app(settings)) as c:
        c.app.state.ollama_admin.client = httpx.AsyncClient(
            transport=httpx.MockTransport(fake_ollama)
        )
        yield c


def test_catalog_endpoint(client):
    body = client.get("/api/catalog", headers=AUTH).json()
    assert body["ollama_online"] is True
    assert body["hardware"]["vram_total_gb"] == 11
    qwen8 = next(v for f in body["families"] for v in f["variants"] if v["tag"] == "qwen3:8b")
    assert qwen8["installed"] and qwen8["fit"] == "gpu"


def test_installed_endpoint(client):
    models = client.get("/api/models/installed", headers=AUTH).json()
    assert models[0]["name"] == "qwen3:8b" and models[0]["size_gb"] == 5.23


def test_pull_and_poll(client):
    resp = client.post("/api/models/pull", headers=AUTH, json={"model": "qwen3:4b"})
    assert resp.status_code == 202
    for _ in range(50):
        pulls = client.get("/api/models/pulls", headers=AUTH).json()
        if pulls and pulls[0]["done"]:
            break
    assert pulls[0]["model"] == "qwen3:4b" and pulls[0]["error"] is None
    assert pulls[0]["status"] == "success"
    assert "qwen3:4b" in WARMED


def test_pull_rejects_bad_name(client):
    resp = client.post("/api/models/pull", headers=AUTH, json={"model": "bad name"})
    assert resp.status_code == 400


def test_delete(client):
    assert client.delete("/api/models/qwen3:8b", headers=AUTH).status_code == 204


def test_model_endpoints_need_auth(client):
    assert client.get("/api/catalog").status_code == 401
    assert client.post("/api/models/pull", json={"model": "x"}).status_code == 401
