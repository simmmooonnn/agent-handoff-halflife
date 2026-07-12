import json, os, tempfile
from typing import Callable

class MockBackend:
    def __init__(self, name: str, fn: Callable[[str], str]):
        self.name = name
        self._fn = fn
    def generate(self, prompt: str, item_id: str, max_tokens: int) -> str:
        return self._fn(prompt)

class CachedBackend:
    def __init__(self, inner, cache_path: str):
        self.inner = inner
        self.name = inner.name
        self.cache_path = cache_path
        self._cache = {}
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
    def generate(self, prompt: str, item_id: str, max_tokens: int) -> str:
        key = f"{self.name}:{item_id}"
        if key in self._cache:
            return self._cache[key]
        out = self.inner.generate(prompt, item_id, max_tokens)
        self._cache[key] = out
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=0)
        os.replace(tmp, self.cache_path)
        return out
