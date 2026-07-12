import json
from backends import MockBackend, CachedBackend

def test_cache_persists_and_skips_recompute(tmp_path):
    calls = {"n": 0}
    def fn(prompt):
        calls["n"] += 1
        return "ECHO:" + prompt[:3]
    cache = tmp_path / "c.json"
    b = CachedBackend(MockBackend("mock", fn), str(cache))
    a1 = b.generate("hello", item_id="x1", max_tokens=8)
    a2 = b.generate("hello", item_id="x1", max_tokens=8)   # same item_id -> cached
    assert a1 == a2 == "ECHO:hel"
    assert calls["n"] == 1                                  # only computed once
    # new process: cache read from disk, no recompute
    b2 = CachedBackend(MockBackend("mock", fn), str(cache))
    assert b2.generate("hello", item_id="x1", max_tokens=8) == a1
    assert calls["n"] == 1
    assert "mock:x1" in json.loads(cache.read_text())
