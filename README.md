# Digit Recognizer

A small convolutional neural network that reads handwritten digits. It's trained from scratch on MNIST with PyTorch (98.9% test accuracy, about 47s on an Apple Silicon GPU) and can then be fine-tuned on your own handwriting.

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python train.py      # train from scratch, saves digit_model.pt
.venv/bin/python predict.py    # watch it guess random test digits
.venv/bin/python draw.py       # draw your own digits at http://localhost:8000
.venv/bin/python finetune.py   # fine-tune on your drawings, saves my_model.pt
```

See [NOTES.md](NOTES.md) for how it works, key concepts, and experiment results.
