# Digit Recognizer

A small convolutional neural network that reads handwritten digits. It's trained from scratch on MNIST with PyTorch (99.3% test accuracy, about 2 min on an Apple Silicon GPU) and can then be fine-tuned on your own handwriting.

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python train.py      # train from scratch, saves digit_model.pt
.venv/bin/python predict.py    # watch it guess random test digits
.venv/bin/python draw.py       # draw your own digits at http://localhost:8000
.venv/bin/python finetune.py   # fine-tune on your drawings, saves my_model.pt
```

See [NOTES.md](NOTES.md) for how it works, key concepts, and experiment results, and [AI Training Methods](<AI Training Methods/TRAINING_METHODS.md>) for an overview of AI training methods in general.

## More experiments
| Folder | What it trains | Method |
|---|---|---|
| [language_model/](language_model/NOTES.md) | A tiny GPT that writes Shakespeare | Self-supervised (how LLMs are pretrained) |
| [autoencoder/](autoencoder/NOTES.md) | A VAE that compresses digits to 2 numbers and draws new ones | Unsupervised / generative |
| [snake_rl/](snake_rl/NOTES.md) | A Snake agent that learns by playing (0.2 → 20 apples per game) | Reinforcement learning (Q-learning) |
