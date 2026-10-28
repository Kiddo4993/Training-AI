"""Export trained models for the browser demo in docs/.

Run:  .venv/bin/python scripts/export_web_demo.py
Writes docs/digitnet.bin (all DigitNet weights as little-endian float32, in state_dict order)
and docs/q_table.json (copy of the trained Snake Q-table).
"""
import shutil
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

state = torch.load(ROOT / "digit_model.pt", map_location="cpu", weights_only=True)
order = ["features.0.weight", "features.0.bias", "features.3.weight", "features.3.bias",
         "classifier.1.weight", "classifier.1.bias", "classifier.4.weight", "classifier.4.bias"]
assert list(state) == order, "DigitNet layout changed: update docs/digitnet.js to match"
blob = np.concatenate([state[k].numpy().astype("<f4").ravel() for k in order])
(DOCS / "digitnet.bin").write_bytes(blob.tobytes())
shutil.copy(ROOT / "snake_rl" / "q_table.json", DOCS / "q_table.json")
print(f"digitnet.bin: {blob.size:,} weights ({blob.nbytes / 1e6:.1f} MB); q_table.json copied")
