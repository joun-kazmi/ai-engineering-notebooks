import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Make `src` importable as a package from the repo root, and `ai_engineering`
# importable directly (as the notebooks do)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Tests are offline by contract. Blank every provider key *before* anything
# imports ai_engineering.config: real env vars beat .env in pydantic-settings,
# and the config treats a blank key as unset, so a developer's exported
# NVIDIA_API_KEY can't silently turn the suite into live (paid, flaky) calls.
for key in ("NVIDIA_API_KEY", "OPENAI_API_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ[key] = ""
os.environ["LLM_PROVIDER"] = "nvidia"

# The served agent persists runs to SQLite; never into the repo from tests.
import tempfile

os.environ["SERVE_DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="serve-tests-")) / "serve.sqlite3")
