# Tiny Language Model (character-level GPT)

*Started 2026-10-20*

## What it is
A small **transformer**, the same architecture as ChatGPT and Claude but about 800k parameters instead of billions. It's trained on ~1MB of Shakespeare and learns to write text that looks like it.

**Training method: self-supervised learning.** No one labels anything. The text labels itself: for every position, the right answer is the next character. That's how real LLMs are pretrained (Section 4 of [AI Training Methods](<../AI Training Methods/TRAINING_METHODS.md>)).

## How to run
```bash
cd ~/Desktop/ai
.venv/bin/python language_model/train_lm.py   # ~7 min, saves lm_model.pt
.venv/bin/python language_model/generate.py   # writes new text (needs sample_next_char)
```

## Key concepts
- **Token**: one unit of text. Here it's one character (65 possible). Real LLMs use word pieces (~100k possible).
- **Context window (`BLOCK_SIZE` = 128)**: how far back the model can look when predicting the next character.
- **Attention**: lets each character look at earlier characters and decide which ones matter (for example, a `"` that hasn't been closed yet).
- **Causal mask**: stops the model from looking ahead at the answer during training.
- **Loss**: starts around 4.17 (ln 65, a random guess among 65 characters). A loss of 1.5 means the model has narrowed it down to about e^1.5 ≈ 4.5 likely characters.
- **Greedy vs. sampling**: always picking the top character gets stuck repeating ("the the the"). Sampling with **temperature** and **top-k** adds controlled randomness.
- **Cosine LR schedule**: the learning rate starts at 1e-3 and smoothly shrinks to 0 by the last step.

## Results
| Run | Steps | Best val loss | Time | Sample |
|---|---|---|---|---|
| 1 (2026-10-21) | 3000 | 1.649 | 344s | "If the treath ensent sir; take it and I say;" |

Run 1: the greedy preview was stuck on "the the the" at step 300 and still looping ("the stand of the stand") at step 3000. Sampling with temperature 0.8 and top-k 20 gives varied text with real character names (CORIOLANUS, LUCIO, ANGELO) and play formatting, though many words are invented. Val loss was still falling, so more steps or a bigger model would help.
