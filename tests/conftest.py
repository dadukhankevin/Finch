"""Tests must never pollute the machine's real run registry — every
test session gets its own throwaway registry file."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path_factory, monkeypatch):
    monkeypatch.setenv(
        "FINCH4_REGISTRY",
        str(tmp_path_factory.getbasetemp() / "registry.jsonl"))
