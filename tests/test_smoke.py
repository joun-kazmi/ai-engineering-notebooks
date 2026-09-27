"""Offline smoke tests: no LLM calls, no real API keys.

Most of these guard against specific setup bugs that previously made the
README path fail for a new user.
"""
import json
import os
import py_compile
import re
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
SOURCES = SCRIPTS + [ROOT / "src" / "fastapi_serve.py"]
NOTEBOOKS = sorted((ROOT / "notebooks").rglob("*.ipynb"))

ENV_VAR = re.compile(r"""os\.(?:environ(?:\.get)?\s*[\[(]|getenv\s*\()\s*["']([A-Z0-9_]+)["']""")


def code_of(path: Path) -> str:
    """Source code of a script, or the joined code cells of a notebook (outputs excluded)."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    return path.read_text(encoding="utf-8")


ALL_CODE = SOURCES + NOTEBOOKS


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


@pytest.mark.parametrize("path", SOURCES, ids=rel)
def test_python_compiles(path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=rel)
def test_notebook_is_valid(path):
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=rel)
def test_notebook_has_no_error_outputs(path):
    # Committed outputs are what readers see; a stack trace there means a broken run was checked in
    nb = nbformat.read(path, as_version=4)
    errors = [
        f"{out.get('ename')}: {out.get('evalue', '')[:80]}"
        for cell in nb.cells
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors, errors


def test_env_example_lists_every_variable_the_code_reads():
    documented = set(re.findall(r"^([A-Z0-9_]+)=", (ROOT / ".env.example").read_text(), re.M))
    used = {name for p in ALL_CODE for name in ENV_VAR.findall(code_of(p))}
    assert used, "found no env var reads; the regex is probably stale"
    assert used <= documented, f"read by code but missing from .env.example: {sorted(used - documented)}"


@pytest.mark.parametrize("path", ALL_CODE, ids=rel)
def test_code_that_reads_env_loads_dotenv(path):
    code = code_of(path)
    if ENV_VAR.search(code):
        assert "load_dotenv()" in code, "reads env vars but never loads .env"


@pytest.mark.parametrize("path", SCRIPTS, ids=rel)
def test_scripts_do_not_use_cwd_relative_data_paths(path):
    assert not re.search(r"""["']\.\./data/""", path.read_text()), "resolve data/ from __file__ instead"


@pytest.mark.parametrize("path", ALL_CODE, ids=rel)
def test_chat_messages_use_content_key(path):
    # {"role": "user", "text": ...} is silently dropped by some endpoints
    assert not re.search(r"""["']role["']\s*:\s*["']\w+["']\s*,\s*["']text["']\s*:""", code_of(path))


# Word-to-word similarity is symmetric, so both sides legitimately use "query" there.
SYMMETRIC_EMBEDDING = {"cosine_similarity"}


@pytest.mark.parametrize("path", ALL_CODE, ids=rel)
def test_documents_are_not_embedded_as_queries(path):
    if path.stem in SYMMETRIC_EMBEDDING:
        pytest.skip("symmetric comparison")
    code = code_of(path)
    assert not re.search(r"""extra_body\s*=\s*\{\s*["']input_type["']\s*:\s*["']query["']""", code), (
        "hardcoded input_type='query' embeds documents as queries; take input_type as a parameter"
    )


@pytest.fixture(scope="module")
def serve():
    # No key needed: the service creates its LLM client lazily, and without a
    # key the agent runs offline (conftest.py blanks every key).
    import src.fastapi_serve as serve
    return serve


def test_readme_serve_command_targets_the_fastapi_app(serve):
    from fastapi import FastAPI

    match = re.search(r"uvicorn src\.fastapi_serve:(\w+)", (ROOT / "README.md").read_text())
    assert match, "README no longer documents the uvicorn command"
    assert isinstance(getattr(serve, match.group(1)), FastAPI)


def test_service_exposes_expected_routes(serve):
    paths = {r.path for r in serve.app_fastapi.routes}
    assert {"/generate", "/alert", "/approve", "/runs/{thread_id}", "/runs/{thread_id}/recover"} <= paths


def test_approve_unknown_thread_returns_404(serve):
    from fastapi.testclient import TestClient

    resp = TestClient(serve.app_fastapi).post(
        "/approve", json={"thread_id": "does-not-exist", "approved": True, "approver": "ci", "args_hash": "x"}
    )
    assert resp.status_code == 404


def test_escalation_graph_compiles_with_all_nodes(serve):
    nodes = set(serve.app.get_graph().nodes)
    assert {"triage", "investigate", "verify", "write_rca", "human_gate", "execute"} <= nodes


def test_sample_data_files_exist():
    for name in ("sample_corpus.txt", "train.jsonl", "val.jsonl"):
        assert (ROOT / "data" / name).is_file(), name
