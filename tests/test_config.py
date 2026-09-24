"""Offline tests for src/ai_engineering/config.py."""
from ai_engineering.config import Settings


def test_blank_key_counts_as_unset(monkeypatch):
    # .env.example ships `NVIDIA_API_KEY=`; copying it must mean OFFLINE, not a 401
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "  ")
    s = Settings(_env_file=None)
    assert s.nvidia_api_key is None
    assert not s.has_llm_credentials
    assert s.resolved_model == "nvidia/nemotron-3-super-120b-a12b"


def test_real_key_and_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    s = Settings(_env_file=None)
    assert s.has_llm_credentials
    assert s.require_api_key() == "sk-test"
    assert s.resolved_model == "gpt-test"
    assert s.resolved_base_url == "https://api.openai.com/v1"
