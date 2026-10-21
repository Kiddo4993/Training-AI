"""Train a tiny character-level GPT on Shakespeare, the same way big LLMs are pretrained.

Run (from ~/Desktop/ai):  .venv/bin/python language_model/train_lm.py
Output: language_model/lm_model.pt (trained weights) + loss and a text preview every few hundred steps.
"""
import time
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).parent
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = HERE / "data" / "shakespeare.txt"

STEPS = 3000
EVAL_EVERY = 300
BATCH_SIZE = 64
BLOCK_SIZE = 128      # how many characters of context the model can look back at
EMBED_DIM = 128       # size of the vector that represents each character
N_HEADS = 4           # attention heads per layer (each can focus on a different pattern)
N_LAYERS = 4
DROPOUT = 0.1
LEARNING_RATE = 1e-3

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. Data -------------------------------------------------------------
# ~1MB of Shakespeare plays. Every character gets an id (65 unique ones).
if not DATA_FILE.exists():
    DATA_FILE.parent.mkdir(exist_ok=True)
    # Download to a temp name and rename at the end, so an interrupted download can't leave a half file behind.
    partial = DATA_FILE.with_suffix(".part")
    urllib.request.urlretrieve(DATA_URL, partial)
    partial.rename(DATA_FILE)
text = DATA_FILE.read_text()
chars = sorted(set(text))
vocab_size = len(chars)
char_to_id = {c: i for i, c in enumerate(chars)}


def encode(s):
    return [char_to_id[c] for c in s]


def decode(ids):
    return "".join(chars[i] for i in ids)


data = torch.tensor(encode(text))
split = int(0.9 * len(data))  # first 90% to train on, last 10% to check generalization
train_data, val_data = data[:split], data[split:]


def get_batch(source):
    # Self-supervised: there are no human labels. The "answer" for each position
    # is simply the next character, so y is x shifted one place to the left.
    starts = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[s : s + BLOCK_SIZE] for s in starts])
    y = torch.stack([source[s + 1 : s + BLOCK_SIZE + 1] for s in starts])
    return x.to(device), y.to(device)


# --- 2. Model ------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, EMBED_DIM)  # which character
        self.pos_embed = nn.Embedding(BLOCK_SIZE, EMBED_DIM)    # where it sits in the sequence
        layer = nn.TransformerEncoderLayer(
            EMBED_DIM, N_HEADS, 4 * EMBED_DIM, DROPOUT, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)            # one score per possible next character

    def forward(self, ids):
        length = ids.shape[1]
        x = self.token_embed(ids) + self.pos_embed(torch.arange(length, device=ids.device))
        # Causal mask: each position may only look at earlier characters, never peek at the answer.
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=ids.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.head(self.norm(x))


def compute_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))


def greedy(logits):
    """Always pick the single highest-scoring character. Safe, but gets stuck repeating itself."""
    return logits.argmax().item()


@torch.no_grad()
def generate(model, prompt, length, pick=greedy):
    """Write `length` new characters after `prompt`, one at a time. `pick` chooses each next character."""
    model.eval()
    ids = torch.tensor([encode(prompt)], device=device)
    for _ in range(length):
        logits = model(ids[:, -BLOCK_SIZE:])[0, -1]  # scores for the character after the last one
        next_id = pick(logits)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    model.train()
    return decode(ids[0].tolist())


# --- 3. Train ------------------------------------------------------------
@torch.no_grad()
def estimate_val_loss(model, batches=20):
    model.eval()
    loss = sum(compute_loss(model, *get_batch(val_data)).item() for _ in range(batches)) / batches
    model.train()
    return loss


if __name__ == "__main__":
    model = TinyGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Cosine schedule: big steps early, gradually smaller ones so training settles at the end.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS)

    print(f"Training on {device}, {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{len(text):,} characters, vocab of {vocab_size}")
    start = time.time()
    best_val = float("inf")
    window_loss = torch.zeros((), device=device)  # sum of train losses since the last eval
    for step in range(1, STEPS + 1):
        loss = compute_loss(model, *get_batch(train_data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        window_loss += loss.detach()

        if step % EVAL_EVERY == 0:
            train_loss = window_loss.item() / EVAL_EVERY  # averaged, so it's comparable to the val loss
            window_loss.zero_()
            val = estimate_val_loss(model)
            marker = ""
            if val < best_val:  # keep only the best checkpoint, like train.py does
                best_val = val
                # Save the character list too, so generate.py can't decode with a different vocabulary.
                torch.save({"model": model.state_dict(), "chars": chars}, HERE / "lm_model.pt")
                marker = "  <- best so far, saved"
            print(f"Step {step}: train loss {train_loss:.3f} | val loss {val:.3f}{marker}")
            print("   greedy preview:", repr(generate(model, "ROMEO:", 60)[6:]))
    print(f"Done in {time.time() - start:.0f}s. Best val loss {best_val:.3f}, saved to {HERE.name}/lm_model.pt")
