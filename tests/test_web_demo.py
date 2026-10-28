"""The browser demo (docs/) must behave exactly like the Python models it was exported from."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from train import DigitNet, test_data

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    images = torch.stack([test_data[i][0] for i in range(20)])
    path = tmp_path_factory.mktemp("demo") / "inputs.json"
    path.write_text(json.dumps(images.view(20, -1).tolist()))
    out = subprocess.run(["node", str(ROOT / "tests" / "web_demo_check.mjs"), str(ROOT / "docs"), str(path)],
                         capture_output=True, text=True, check=True)
    return images, json.loads(out.stdout)


def test_exported_weights_are_up_to_date():
    state = torch.load(ROOT / "digit_model.pt", map_location="cpu", weights_only=True)
    exported = torch.frombuffer(bytearray((ROOT / "docs" / "digitnet.bin").read_bytes()), dtype=torch.float32)
    assert torch.equal(exported, torch.cat([t.flatten() for t in state.values()])), \
        "docs/digitnet.bin is stale: run scripts/export_web_demo.py"


def test_javascript_network_matches_pytorch(demo):
    images, result = demo
    model = DigitNet()
    model.load_state_dict(torch.load(ROOT / "digit_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        expected = model(images)
    assert torch.allclose(torch.tensor(result["logits"]), expected, atol=1e-3)


def test_javascript_snake_plays_like_the_python_agent(demo):
    _, result = demo
    assert result["trained"] >= 10 and result["random"] < 1
