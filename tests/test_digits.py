"""Digit recognizer (train.py) and handwriting fine-tuning (finetune.py)."""
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from train import DigitNet, device, test_loader
from finetune import configure_finetuning, to_mnist_style

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def mnist_accuracy(weights):
    model = DigitNet().to(device)
    model.load_state_dict(torch.load(ROOT / weights, map_location=device, weights_only=True))
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        correct += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
        total += len(y)
    return correct / total


def test_model_outputs_one_score_per_digit():
    assert DigitNet()(torch.zeros(4, 1, 28, 28)).shape == (4, 10)


def test_trained_model_hits_99_percent():
    assert mnist_accuracy("digit_model.pt") >= 0.99


@pytest.mark.skipif(not (ROOT / "my_model.pt").exists(), reason="no fine-tuned model")
def test_fine_tuned_model_still_reads_mnist():
    assert mnist_accuracy("my_model.pt") >= 0.98


def test_drawing_is_converted_to_mnist_style(tmp_path):
    img = Image.new("RGBA", (280, 280))
    ImageDraw.Draw(img).line([(40, 30), (60, 200)], fill="white", width=22)  # off-center stroke
    img.save(tmp_path / "one.png")
    out = to_mnist_style(tmp_path / "one.png")
    assert out.size == (28, 28)
    assert max(out.getbbox()[2] - out.getbbox()[0], out.getbbox()[3] - out.getbbox()[1]) <= 21


def test_blank_drawing_is_skipped(tmp_path):
    Image.new("RGBA", (280, 280)).save(tmp_path / "blank.png")
    assert to_mnist_style(tmp_path / "blank.png") is None


def test_fine_tuning_freezes_conv_layers():
    model = DigitNet()
    optimizer = configure_finetuning(model)
    assert not any(p.requires_grad for p in model.features.parameters())
    trained = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert trained == {id(p) for p in model.classifier.parameters()}
