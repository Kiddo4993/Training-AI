# Training an AI: Handwritten Digit Recognizer

*Started 2026-10-07*

## What I built
A **convolutional neural network (CNN)** that looks at a 28×28 picture of a handwritten digit and says which digit (0–9) it is. It was trained on **MNIST**, a classic dataset of 70,000 handwritten digits (60k for training, 10k for testing).

## Results (first run)
| Epoch | Train loss | Test accuracy |
|---|---|---|
| 1 | 0.1574 | 98.45% |
| 2 | 0.0537 | 99.04% |
| 3 | 0.0387 | 98.87% |

- Trained in **47 seconds** on the Mac GPU (`mps`)
- Model size: **421,642 parameters** (the numbers the AI learns)
- Accuracy dipped slightly in epoch 3 even though train loss kept falling. That's a small hint of **overfitting**: the model memorizes the training set a bit instead of getting better at new digits.

## Files in this folder
| File | What it is |
|---|---|
| `train.py` | Downloads data, builds the model, trains it, saves weights |
| `predict.py` | Loads the trained model and shows it guessing random test digits (drawn in ASCII) |
| `draw.py` | Browser drawing pad (localhost:8000) to collect my own digits |
| `finetune.py` | Fine-tunes the model on my drawings → `my_model.pt` |
| `digit_model.pt` | The trained weights. This file *is* the trained AI |
| `data/` | The downloaded MNIST dataset |
| `.venv/` | Python environment with PyTorch installed |

## How to run
```bash
cd ~/Desktop/ai
.venv/bin/python train.py     # train again (~1 min)
.venv/bin/python predict.py   # watch it guess
```

## Key concepts

### What "training" actually means
A neural network is a big math function full of adjustable numbers (**weights**). At the start they're random, so the guesses are random too. Training repeats this loop thousands of times:

1. **Forward pass**: show the model a batch of 64 images and get its guesses
2. **Loss**: measure how wrong the guesses were (cross-entropy loss)
3. **Backpropagation** (`loss.backward()`): use calculus (the chain rule) to work out which direction to move each weight to reduce the loss
4. **Optimizer step** (`optimizer.step()`): nudge every weight a little in that direction

That's **gradient descent**: walking downhill on the "wrongness" landscape.

### Vocabulary
- **Epoch**: one full pass through all 60,000 training images
- **Batch size (64)**: how many images the model sees before each weight update
- **Learning rate (0.001)**: how big each nudge is. Too big and training overshoots; too small and it's slow
- **Train vs. test set**: the model never trains on the test images, so test accuracy shows how well it *generalizes* to digits it hasn't seen
- **Overfitting**: doing great on training data but worse on new data
- **Optimizer (Adam)**: a smarter version of gradient descent that adapts the step size for each weight

### The model's layers
```
Image 1×28×28
 → Conv 3×3 (32 filters) + ReLU   finds edges and strokes
 → MaxPool                        shrink to 14×14
 → Conv 3×3 (64 filters) + ReLU   combines strokes into shapes (loops, corners)
 → MaxPool                        shrink to 7×7
 → Flatten → Linear 128 + ReLU    reasons about which shapes appear together
 → Dropout 25%                    randomly turns neurons off while training to reduce overfitting
 → Linear 10                      one score per digit; the highest score is the guess
```
- **Convolution**: slides a small filter over the image to find a pattern anywhere it appears
- **ReLU**: `max(0, x)`. Adds the non-linearity that lets the network learn more than straight lines
- **Pooling**: keeps the strongest signal in each 2×2 area, which makes the model smaller and less sensitive to exact position
- **Normalization**: input pixels are shifted so their average is 0, which helps training settle faster

## From scratch vs. starting from an existing model

**Rule of thumb:** a simple task with lots of data means train from scratch. A complex task, or only a little data, means start from a pretrained model (**transfer learning**).

| | From scratch | Starting from a pretrained model |
|---|---|---|
| **Best when** | You have lots of data and a simple or unusual task | You have little data or a complex task (photos, speech, text) |
| **This project** | ✅ 60k digits, 99% in 47s | ❌ Common pretrained models (e.g. ResNet) expect 224×224 color photos. They know textures and objects, not pen strokes |
| **Cost** | More data and compute | Much less of both |

**Why reusing a model works:** early layers learn *general* features (edges, curves) and later layers learn *task-specific* decisions, so you can keep the early layers and retrain only the end.

**Key ideas**
- **Fine-tuning**: keep training an already-trained model on new, usually smaller, data
- **Freezing**: locking a layer's weights (`requires_grad = False`) so training can't change them
- **Small learning rate**: fine-tuning usually uses a rate 10–100× smaller than training from scratch, so it *adjusts* what the model already knows instead of overwriting it
- **Catastrophic forgetting**: fine-tuning too hard on new data makes the model worse at the original task. `finetune.py` checks this by re-scoring MNIST afterwards

## Fine-tuning experiment: teaching it MY handwriting
The MNIST model has never seen my handwriting. 50–100 drawings aren't enough to train from scratch, but they are enough to **fine-tune** `digit_model.pt`.

```bash
.venv/bin/python draw.py       # open http://localhost:8000, draw ~10 of each digit
.venv/bin/python finetune.py   # prints BEFORE vs AFTER accuracy, saves my_model.pt
```

**How it works**
1. `draw.py` saves each drawing to `my_digits/<digit>/NNN.png`
2. `finetune.py` converts each drawing to MNIST style: crop to the ink, fit it in a 20×20 box, then center it by *center of mass* on a 28×28 canvas. Without this, the model sees images that look nothing like its training data
3. 25% of my drawings are held out so the before/after test is fair
4. **Augmentation** (small random rotations, shifts and resizes) makes each drawing count as many slightly different examples
5. It reports accuracy on my handwriting **and** on MNIST, to check for forgetting

**Test run with fake "slanted, thick" handwriting** (Claude's dry run, 80 train / 20 test, freezing the conv layers with lr 1e-4):
| Epochs | My handwriting before → after | MNIST after |
|---|---|---|
| 30 | 80% → 85% | 98.93% |
| 200 | 80% → 90% | 98.80% |

Only 20 test images means each one is worth 5%, so these numbers are noisy. More drawings give more trustworthy results.

**My job:** fill in `configure_finetuning()` in `finetune.py`, deciding *what* to freeze and *how fast* to learn.

| Strategy | What trains | Trade-off |
|---|---|---|
| Freeze conv layers | Classifier only | Safest, least forgetting, but can't learn new stroke shapes |
| Freeze nothing, tiny lr | Everything, gently | More flexible, higher risk of forgetting |
| Freeze nothing, big lr (1e-3) | Everything, hard | Likely overfits 80 drawings and forgets MNIST. Good to try once to *see* it fail |

| Strategy I tried | lr | My handwriting before → after | MNIST after |
|---|---|---|---|
| | | | |

## Experiments to try next
- [ ] Change `EPOCHS` to 1, then to 10. Does accuracy keep going up?
- [ ] Set `LEARNING_RATE` to `0.1`. What breaks?
- [ ] Delete the Conv layers and use only Linear layers. How much worse is it?
- [ ] Remove `Dropout`. Does overfitting get worse?
- [ ] Try `datasets.FashionMNIST` instead (clothing pictures, a harder task)
- [ ] Draw your own digit and have the model read it

## My notes
<!-- add your own observations here -->
