import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import catalog
from app.catalog import live
from app.config import Settings
from app.hardware import Gpu, Hardware
from app.main import create_app

AUTH = {"Authorization": "Bearer k"}
GPU_11GB = Hardware(gpus=[Gpu("2080 Ti", 11, 10)], ram_total_gb=32, disk_free_gb=500)


def fam(name, sizes, use_cases=("chat",), popularity=0, **extra):
    return {
        "id": name,
        "name": name,
        "engine": "ollama",
        "use_cases": list(use_cases),
        "popularity": popularity,
        "variants": [{"tag": f"{name}:{s}", "params": f"{s}B", "size_gb": s} for s in sizes],
        **extra,
    }


def annotated(*families):
    return [catalog.annotate(f, GPU_11GB, set(), set()) for f in families]


# ---- perfect match ----


def test_perfect_match_prefers_the_biggest_model_that_fits():
    [f] = annotated(fam("qwen", [0.5, 2.5, 5.2, 9.3, 20]))
    best = [v["tag"] for v in f["variants"] if v["best"]]
    assert best == ["qwen:5.2"]  # 9.3 GB needs ~12 GB with overhead -> partial


def tags(result):
    return [m["tag"] for m in result["models"]]


def test_perfect_sort_is_one_flat_list_biggest_fitting_first():
    # The Gemma case: 1B and 4B fit, so 4B must come first - across all families.
    models = annotated(
        fam("gemma3", [0.8, 3.3, 8.1, 17.4]),
        fam("tiny", [0.4], popularity=10**9),
        fam("huge", [40]),
    )
    order = tags(catalog.browse(models, sort="perfect"))
    # Bigger models that fit come first, even against an absurdly popular tiny one...
    assert order[:2] == ["gemma3:8.1", "gemma3:3.3"]
    # ...but among tiny models (similar size), the far more popular one wins.
    assert order.index("tiny:0.4") < order.index("gemma3:0.8")
    assert order.index("gemma3:17.4") > order.index("tiny:0.4")  # partial after GPU fits
    assert order[-1] == "huge:40"


def test_perfect_match_weighs_popularity_and_age_between_similar_sizes():
    import time

    now = time.time()
    year = 365 * 86400
    old = fam("qwen", [9.0], source="ollama")
    old["variants"][0].update(downloads=8_000_000, updated=now - 2 * year)
    new = fam("gemma3", [8.2], source="ollama")
    new["variants"][0].update(downloads=41_000_000, updated=now - 30 * 86400)
    tiny = fam("tiny", [0.4], source="ollama")
    tiny["variants"][0].update(downloads=500_000_000, updated=now)
    hw = Hardware(gpus=[Gpu("big", 16, 15)], ram_total_gb=64, disk_free_gb=500)
    rows = [catalog.annotate(f, hw, set(), set()) for f in (old, new, tiny)]
    order = tags(catalog.browse(rows, sort="perfect"))
    # The current, popular 12B-class model beats the slightly bigger two-year-old one...
    assert order[0] == "gemma3:8.2"
    # ...and even a hugely popular tiny model doesn't beat models that are much more capable.
    assert order[-1] == "tiny:0.4"


@pytest.mark.parametrize(
    ("text", "days"),
    [
        ("yesterday", 1),
        ("2 years ago", 730.5),
        ("1 month ago", 30.4),
        ("a year ago", 365.25),
        ("6 days ago", 6),
        ("bogus", None),
    ],
)
def test_age_days(text, days):
    assert live.age_days(text) == days


def test_other_sorts():
    models = annotated(fam("tiny", [0.4], popularity=10**9), fam("mid", [5.0]), fam("huge", [40]))
    assert tags(catalog.browse(models, sort="fit"))[0] == "tiny:0.4"  # most headroom
    assert tags(catalog.browse(models, sort="name")) == ["huge:40", "mid:5.0", "tiny:0.4"]
    assert tags(catalog.browse(models, sort="size_desc"))[0] == "huge:40"
    assert tags(catalog.browse(models, sort="size_asc"))[0] == "tiny:0.4"
    assert tags(catalog.browse(models, sort="popular"))[0] == "tiny:0.4"


def test_browse_filters():
    models = annotated(
        fam("coder", [4], use_cases=("chat", "code"), description="for code"),
        fam("embed", [0.3], use_cases=("embedding",)),
        fam("giant", [200]),
    )
    assert tags(catalog.browse(models, category="code")) == ["coder:4"]
    assert catalog.browse(models, query="EMBED")["total"] == 1
    assert "giant:200" not in tags(catalog.browse(models, runnable_only=True))
    assert catalog.browse(models, limit=1, offset=1)["total"] == 3


def test_rows_carry_family_info_and_unsupported_families():
    cloud = fam("cloudy", [])
    cloud["unsupported"] = "cloud only"
    rows = catalog.flatten(annotated(fam("q", [1.0], description="d"), cloud))
    assert rows[0]["name"] == "q" and rows[0]["description"] == "d" and rows[0]["fit"] == "gpu"
    assert rows[1]["tag"] == "cloudy" and rows[1]["unsupported"]
    assert tags(catalog.browse(annotated(fam("q", [1.0]), cloud)))[-1] == "cloudy"


def test_gated_models_are_never_the_pick_without_a_token():
    f = fam("img", [6.0])
    f["variants"][0]["requires_token"] = True
    [a] = annotated(f)
    assert not a["variants"][0]["best"]


def test_download_counts_per_model_or_family():
    live_ollama = fam("qwen3", [5.2], popularity=40_000_000, source="ollama")
    live_ollama["variants"][0].update(
        tag="qwen3:14b", downloads=40_000_000, downloads_scope="family"
    )
    live_hf = fam("sdxl", [7.0], source="huggingface")
    live_hf["variants"][0].update(tag="org/sdxl", downloads=4_300_000, downloads_scope="model")
    curated = [fam("Qwen 3", [5.2]), fam("SDXL", [7.0])]
    curated[0]["variants"][0]["tag"] = "qwen3:8b"
    curated[1]["variants"][0]["tag"] = "org/sdxl"
    rows = {r["tag"]: r for r in catalog.flatten(catalog.merge(curated, [live_ollama, live_hf]))}
    assert (rows["qwen3:8b"]["downloads"], rows["qwen3:8b"]["downloads_scope"]) == (
        40_000_000,
        "family",
    )
    assert (rows["org/sdxl"]["downloads"], rows["org/sdxl"]["downloads_scope"]) == (
        4_300_000,
        "model",
    )


def test_merge_drops_live_duplicates_of_curated_tags():
    curated = [fam("qwen3", [5.2])]
    curated[0]["variants"][0]["tag"] = "qwen3:8b"
    merged = catalog.merge(curated, [fam("qwen3live", [5.2, 9.3])])
    tags = [v["tag"] for f in merged for v in f["variants"]]
    assert tags.count("qwen3:8b") == 1


# ---- Ollama library parsing ----

LIBRARY_HTML = """
<ul>
<li><a href="/library/qwen3" class="group">
  <p class="max-w-lg break-words">Qwen3 does many things, including code.</p>
  <span class="inline-flex rounded-md bg-indigo-50 text-indigo-600">tools</span>
  <span class="inline-flex rounded-md bg-indigo-50 text-indigo-600">thinking</span>
  <span class="inline-flex rounded-md bg-[#ddf4ff] text-blue-600">0.6b</span>
  <span class="inline-flex rounded-md bg-[#ddf4ff] text-blue-600">8b</span>
  <p><span>39.6M</span> <span class="hidden sm:flex">&nbsp;Pulls</span></p>
</a></li>
<li><a href="/library/bigcloud" class="group">
  <p class="max-w-lg">Huge.</p>
  <span class="inline-flex rounded-md bg-cyan-50 text-cyan-500">cloud</span>
  <p><span>1.2K</span> <span class="hidden sm:flex">&nbsp;Pulls</span></p>
</a></li>
<li><a href="/library/nomic-embed-text" class="group">
  <p class="max-w-lg">Embeddings.</p>
  <span class="inline-flex rounded-md bg-indigo-50 text-indigo-600">embedding</span>
  <p><span>88M</span> <span class="hidden sm:flex">&nbsp;Pulls</span></p>
</a></li>
</ul>
"""


def test_parse_ollama_library():
    models = {m["name"]: m for m in live.parse_ollama_library(LIBRARY_HTML)}
    q = models["qwen3"]
    assert q["capabilities"] == ["tools", "thinking"] and q["sizes"] == ["0.6b", "8b"]
    assert q["pulls"] == 39_600_000
    fq = live.ollama_family(q, {"qwen3:8b": 5.23})
    # Mentioning code in passing doesn't make it a code model...
    assert fq["use_cases"] == ["chat", "reasoning", "agent"] and fq["tools"]
    # ...but the name (or an explicit "code generation" description) does.
    coder = live.ollama_family({**q, "name": "qwen2.5-coder"}, {})
    assert "code" in coder["use_cases"]
    assert [v["size_gb"] for v in fq["variants"]] == [0.56, 5.23]
    assert fq["variants"][1]["size_estimated"] is False
    cloud = live.ollama_family(models["bigcloud"], {})
    assert cloud["unsupported"] and cloud["variants"] == []
    emb = live.ollama_family(models["nomic-embed-text"], {})
    assert (
        emb["use_cases"] == ["embedding"] and emb["variants"][0]["tag"] == "nomic-embed-text:latest"
    )


@pytest.mark.parametrize(
    ("label", "params"), [("8b", 8), ("137m", 0.137), ("8x7b", 56), ("e4b", 4), ("latest", None)]
)
def test_params_from_label(label, params):
    assert live.params_from_label(label) == params


# ---- Hugging Face ----


def test_hf_family_filters_and_gating():
    model = {
        "id": "black-forest-labs/FLUX.1-dev",
        "tags": ["diffusers:FluxPipeline"],
        "gated": "auto",
        "downloads": 5,
        "safetensors": {"total": 12_000_000_000},
    }
    f = live.hf_family(model, "image", hf_token_set=False)
    assert f["variants"][0]["requires_token"] and f["variants"][0]["size_gb"] == 24.0
    assert (
        live.hf_family(model, "image", hf_token_set=True)["variants"][0]["requires_token"] is False
    )
    assert live.hf_family({"id": "x/y-GGUF", "tags": ["gguf"]}, "image", False) is None
    tts = live.hf_family({"id": "a/tts", "tags": []}, "speech", False)
    assert tts["unsupported"]


def test_diffusers_spec_picks_fp16_files_and_sizes_components():
    siblings = [
        {"rfilename": "model_index.json", "size": 500},
        {"rfilename": "unet/config.json", "size": 1},
        {"rfilename": "unet/diffusion_pytorch_model.safetensors", "size": 10_000_000_000},
        {"rfilename": "unet/diffusion_pytorch_model.fp16.safetensors", "size": 5_000_000_000},
        {"rfilename": "text_encoder/model.safetensors", "size": 2_000_000_000},
        {"rfilename": "text_encoder/model.fp16.safetensors", "size": 1_000_000_000},
        {"rfilename": "vae/diffusion_pytorch_model.fp16.safetensors", "size": 100_000_000},
        {"rfilename": "sd_xl_base.safetensors", "size": 7_000_000_000},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/models/org/sdxl":
            return httpx.Response(
                200,
                json={
                    "tags": ["diffusers:StableDiffusionXLPipeline"],
                    "siblings": siblings,
                    "pipeline_tag": "text-to-image",
                },
            )
        if path.endswith("/model_index.json"):
            return httpx.Response(
                200,
                json={
                    "_class_name": "X",
                    "unet": ["diffusers", "U"],
                    "text_encoder": ["transformers", "T"],
                    "vae": ["diffusers", "V"],
                    "safety_checker": [None, None],
                },
            )
        if path.endswith("/unet/config.json"):
            return httpx.Response(200, json={"sample_size": 128})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    spec = asyncio.run(live.diffusers_spec("org/sdxl", client, None))
    cfg = spec["engine_config"]
    assert cfg["variant"] == "fp16" and cfg["dtype"] == "float16" and cfg["size"] == 1024
    allow = cfg["source"]["allow_patterns"]
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in allow
    assert (
        "unet/diffusion_pytorch_model.safetensors" not in allow
        and "sd_xl_base.safetensors" not in allow
    )
    assert cfg["components_gb"] == {"unet": 5.0, "text_encoder": 1.0, "vae": 0.1}
    assert spec["size_gb"] == 6.1


def test_diffusers_spec_gated():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(PermissionError):
        asyncio.run(live.diffusers_spec("org/gated", client, None))


# ---- settings API ----


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, api_key="k", ollama_url="", static_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        yield c


def test_hf_token_lifecycle(client):
    def hf(request: httpx.Request) -> httpx.Response:
        ok = request.headers.get("authorization") == "Bearer hf_good_token"
        return httpx.Response(200, json={"name": "alex"}) if ok else httpx.Response(401)

    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(hf))
    assert client.get("/api/settings", headers=AUTH).json()["huggingface"]["token_set"] is False
    bad = client.put("/api/settings/huggingface", headers=AUTH, json={"token": "hf_bad_token"})
    assert bad.status_code == 400 and "rejected" in bad.json()["detail"]
    ok = client.put(
        "/api/settings/huggingface", headers=AUTH, json={"token": "hf_good_token"}
    ).json()
    assert ok == {"token_set": True, "username": "alex", "source": "ui"}
    assert client.app.state.user_settings.hf_token == "hf_good_token"
    secret = client.app.state.settings.data_dir / "secrets" / "huggingface.json"
    assert json.loads(secret.read_text())["token"] == "hf_good_token"
    assert "hf_good_token" not in client.get("/api/settings", headers=AUTH).text
    assert client.delete("/api/settings/huggingface", headers=AUTH).json()["token_set"] is False


def test_env_token_is_read_only(tmp_path):
    settings = Settings(
        data_dir=tmp_path, api_key="k", ollama_url="", static_dir=tmp_path, hf_token="hf_env"
    )
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/settings", headers=AUTH).json()["huggingface"]["source"] == "env"
        assert c.delete("/api/settings/huggingface", headers=AUTH).status_code == 409


def test_whisper_sizes_come_from_the_name():
    from app.catalog.live import whisper_size_gb

    assert whisper_size_gb("Systran/faster-whisper-large-v3") == 3.1
    assert whisper_size_gb("rhasspy/faster-whisper-small-int8") == 0.24
    assert whisper_size_gb("deepdml/faster-whisper-large-v3-turbo-ct2") == 1.62
    assert whisper_size_gb("Systran/faster-distil-whisper-medium.en") == 0.79
    assert whisper_size_gb("Sunbird/faster-whisper-51-african-languages") is None


def test_sizes_the_listing_lacks_are_added_up_from_the_files(tmp_path):
    import asyncio

    import httpx

    from app.catalog.live import LiveCatalog

    def hf(request):
        if request.url.path == "/api/models":
            return httpx.Response(
                200,
                json=[
                    {"id": "Sunbird/faster-whisper-51-african-languages", "tags": ["ctranslate2"]}
                ],
            )
        siblings = [
            {"rfilename": "model.bin", "size": 3_087_000_000},
            {"rfilename": "README.md", "size": 5000},
        ]
        return httpx.Response(200, json={"siblings": siblings})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(hf)) as c:
            return await LiveCatalog(tmp_path, c).hf_families("transcription")

    variant = asyncio.run(go())[0]["variants"][0]
    assert variant["size_gb"] == 3.09 and variant["vram_gb"] == 4.6
