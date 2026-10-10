from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_client(tmp_path):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>app</html>")
    (static / "assets" / "index-abc123.js").write_text("console.log(1)")
    settings = Settings(data_dir=tmp_path, api_key="k", ollama_url="", static_dir=static)
    return TestClient(create_app(settings))


def test_index_is_revalidated(tmp_path):
    with make_client(tmp_path) as c:
        for path in ("/", "/some/client/route"):
            r = c.get(path)
            assert r.text == "<html>app</html>"
            assert r.headers["cache-control"] == "no-cache"


def test_hashed_assets_are_immutable(tmp_path):
    with make_client(tmp_path) as c:
        r = c.get("/assets/index-abc123.js")
        assert r.status_code == 200
        assert "immutable" in r.headers["cache-control"]
        assert c.get("/assets/missing.js").status_code == 404
