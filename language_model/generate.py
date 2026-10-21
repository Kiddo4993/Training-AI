"""Load the trained tiny GPT and have it write new "Shakespeare".

Run (from ~/Desktop/ai):  .venv/bin/python language_model/generate.py
"""
import torch

from train_lm import HERE, TinyGPT, chars, device, generate

PROMPT = "ROMEO:"
LENGTH = 500
TEMPERATURE = 0.8   # <1 = safer and more repetitive, >1 = wilder and more random
TOP_K = 20          # only allow the 20 most likely characters (None = allow all)


def sample_next_char(logits, temperature=TEMPERATURE, top_k=TOP_K):
    """Choose the next character id from the model's scores.

    logits: 1-D tensor with one score per character (65 of them). Higher = the model thinks it's more likely.
    Return a plain Python int (the chosen character id).

    train_lm.py uses greedy() (always the top score), which loops forever on phrases like
    "the the the". Sampling randomly *in proportion to* the probabilities fixes that.
    """
    if temperature <= 0:  # temperature 0 means "no randomness", which is just greedy
        return logits.argmax().item()
    logits = logits / temperature
    if top_k is not None:
        top_k = min(top_k, logits.numel())  # can't keep more characters than exist
        cutoff = torch.topk(logits, top_k).values[-1]  # score of the k-th best character
        logits = logits.masked_fill(logits < cutoff, float("-inf"))  # -inf becomes probability 0
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1).item()  # random draw, weighted by probability


checkpoint = torch.load(HERE / "lm_model.pt", map_location=device, weights_only=True)
if checkpoint["chars"] != chars:
    raise SystemExit("shakespeare.txt has changed since training, so the characters no longer match. "
                     "Re-run train_lm.py.")
missing = sorted(set(PROMPT) - set(chars))
if not PROMPT or missing:
    raise SystemExit(f"PROMPT must be non-empty and only use characters from the training text (bad: {missing}).")

model = TinyGPT().to(device)
model.load_state_dict(checkpoint["model"])
print(generate(model, PROMPT, LENGTH, pick=sample_next_char))
