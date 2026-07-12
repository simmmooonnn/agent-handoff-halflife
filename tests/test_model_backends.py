import pytest
from model_backends import build_model_backend

def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_model_backend("nope", "x")

def test_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_model_backend("openai", "gpt-4o")

def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_model_backend("anthropic", "claude-sonnet-4-6")
