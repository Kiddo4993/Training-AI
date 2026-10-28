"""Make every project folder importable, the same way each script runs from the repo root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for folder in (ROOT, ROOT / "snake_rl", ROOT / "snake_dqn", ROOT / "language_model", ROOT / "autoencoder"):
    sys.path.insert(0, str(folder))
