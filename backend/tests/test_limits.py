from app.catalog import flatten
from app.catalog.limits import limits_for


def kinds(family: dict) -> list[str]:
    return [lim["kind"] for lim in limits_for(family)]


def test_known_families_have_limits():
    assert kinds({"id": "llama3", "name": "Llama 3.1"}) == ["content", "license"]
    assert "content" in kinds({"id": "ollama:gemma3", "name": "gemma3"})
    texts = [lim["text"] for lim in limits_for({"id": "qwen3", "name": "Qwen3"})]
    assert any("Chinese" in t for t in texts)


def test_source_prefix_does_not_match():
    # "ollama:" contains "llama"; phi4 isn't a Llama model.
    assert all("Llama" not in lim["text"] for lim in limits_for({"id": "ollama:phi4"}))


def test_uncensored_variants_drop_refusals():
    lims = limits_for({"id": "ollama:llama2-uncensored", "name": "llama2-uncensored"})
    assert [lim["kind"] for lim in lims] == ["open", "license"]


def test_license_and_tags_are_read():
    assert kinds({"id": "hf:a/b", "name": "b", "license": "cc-by-nc-4.0"}) == ["license"]
    assert kinds({"id": "hf:a/c", "name": "c", "tags": ["not-for-all-audiences"]}) == ["open"]
    assert kinds(
        {"id": "embeddinggemma", "name": "EmbeddingGemma", "use_cases": ["embedding"]}
    ) == ["license"]
    assert limits_for({"id": "nomic-embed-text", "name": "Nomic Embed"}) == []


def test_rows_carry_limits():
    rows = flatten(
        [{"id": "gpt-oss", "name": "gpt-oss", "use_cases": ["chat"], "variants": [{"tag": "x"}]}]
    )
    assert rows[0]["limits"][0]["kind"] == "content"
