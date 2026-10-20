"""Fine-tune the trained digit model on YOUR handwriting (transfer learning).

1. Draw digits:  .venv/bin/python draw.py   (saves to my_digits/)
2. Fine-tune:    .venv/bin/python finetune.py
Output: my_model.pt + before/after accuracy on your held-out drawings and on MNIST.
"""
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from train import DigitNet, device, test_loader

DIGITS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "my_digits"
EPOCHS = 200  # each epoch = one update over all your drawings (tiny dataset)
HOLDOUT = 0.25  # fraction of your drawings kept aside to test on

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # same normalization as train.py
])
# Small random wiggles so 50 drawings act like many more.
augment = transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))


def to_mnist_style(path):
    """Make a 280x280 drawing look like an MNIST image: digit fit in a 20x20 box,
    centered by center of mass on a 28x28 black background."""
    ink = Image.open(path).convert("RGBA").getchannel("A")  # stroke opacity = ink
    bbox = ink.getbbox()
    if bbox is None:
        return None  # blank drawing
    digit = ink.crop(bbox)
    scale = 20 / max(digit.size)
    digit = digit.resize((max(1, round(digit.width * scale)), max(1, round(digit.height * scale))),
                         Image.LANCZOS)

    t = transforms.functional.pil_to_tensor(digit)[0].float()
    ys, xs = torch.meshgrid(torch.arange(t.shape[0]), torch.arange(t.shape[1]), indexing="ij")
    cy, cx = (t * ys).sum() / t.sum(), (t * xs).sum() / t.sum()

    canvas = Image.new("L", (28, 28))
    canvas.paste(digit, (round(14 - cx.item()), round(14 - cy.item())))
    return canvas


def load_split():
    rng = random.Random(0)
    train, test = [], []
    for d in range(10):
        imgs = [img for p in sorted((DIGITS_DIR / str(d)).glob("*.png"))
                if (img := to_mnist_style(p)) is not None]
        rng.shuffle(imgs)
        n_test = round(len(imgs) * HOLDOUT) if len(imgs) >= 2 else 0
        test += [(img, d) for img in imgs[:n_test]]
        train += [(img, d) for img in imgs[n_test:]]
    return train, test


@torch.no_grad()
def accuracy_on(model, samples):
    if not samples:
        return float("nan")
    model.eval()
    x = torch.stack([to_tensor(img) for img, _ in samples]).to(device)
    y = torch.tensor([d for _, d in samples], device=device)
    return (model(x).argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def mnist_accuracy(model):
    """Did fine-tuning make it forget regular MNIST digits? (catastrophic forgetting)"""
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += len(y)
    return correct / total


def configure_finetuning(model):
    """Decide WHAT gets retrained and HOW FAST. Return an optimizer.

    model.features   = conv layers (learned strokes/shapes from 60k MNIST digits)
    model.classifier = linear layers (turn those shapes into a digit guess)

    To freeze a layer:  for p in model.features.parameters(): p.requires_grad = False
    Then only give the optimizer the parameters that still train, e.g.
        torch.optim.Adam(model.classifier.parameters(), lr=...)
    (train.py used lr=1e-3 from scratch.)
    """
    # Freeze the conv layers: their stroke/shape detectors already work, and with only
    # ~80 drawings, retraining them would overfit and forget MNIST. Retrain just the
    # classifier, with a learning rate 10x smaller than training from scratch.
    for p in model.features.parameters():
        p.requires_grad = False
    return torch.optim.Adam(model.classifier.parameters(), lr=1e-4)


if __name__ == "__main__":
    train, test = load_split()
    print(f"Your drawings: {len(train)} to train on, {len(test)} held out to test")
    if not train:
        sys.exit("No drawings found. Run draw.py first and save some digits.")

    model = DigitNet().to(device)
    model.load_state_dict(torch.load("digit_model.pt", map_location=device, weights_only=True))
    print(f"BEFORE  your handwriting: {accuracy_on(model, test):.1%} | MNIST: {mnist_accuracy(model):.2%}")

    optimizer = configure_finetuning(model)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch = random.sample(train, len(train))
        x = torch.stack([to_tensor(augment(img)) for img, _ in batch]).to(device)
        y = torch.tensor([d for _, d in batch], device=device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"  epoch {epoch}: loss {loss.item():.4f}")

    print(f"AFTER   your handwriting: {accuracy_on(model, test):.1%} | MNIST: {mnist_accuracy(model):.2%}")
    torch.save(model.state_dict(), "my_model.pt")
    print("Saved fine-tuned weights to my_model.pt")
