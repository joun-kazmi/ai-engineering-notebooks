import sys
from pathlib import Path

# Make `src` importable as a package from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
