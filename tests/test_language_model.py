"""Tiny character-level GPT (language_model/)."""
from pathlib import Path

import torch

from train_lm import BLOCK_SIZE, TinyGPT, chars, device, estimate_val_loss, generate, vocab_size

CHECKPOINT = Path(__file__).resolve().parent.parent / "language_model" / "lm_model.pt"


def test_model_predicts_a_score_for_every_character():
    out = TinyGPT()(torch.zeros(2, BLOCK_SIZE, dtype=torch.long))
    assert out.shape == (2, BLOCK_SIZE, vocab_size)


def test_trained_model_beats_random_guessing():
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    assert checkpoint["chars"] == chars
    model = TinyGPT().to(device)
    model.load_state_dict(checkpoint["model"])
    torch.manual_seed(0)
    # Random guessing over 65 characters scores ln(65) = 4.17. Training reached ~1.65.
    assert estimate_val_loss(model) < 2.0
    text = generate(model, "ROMEO:", 50)
    assert text.startswith("ROMEO:") and len(text) == 56
