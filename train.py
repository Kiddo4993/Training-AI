"""Train a small convolutional neural network to recognize handwritten digits (MNIST).

Run:  .venv/bin/python train.py
Output: digit_model.pt (trained weights) + printed accuracy per epoch.
"""
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Use the Mac's GPU (Metal Performance Shaders) if available.
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. Data -------------------------------------------------------------
# ToTensor turns a 28x28 image into numbers 0..1; Normalize centers them
# around 0 using MNIST's known mean/std, which helps training converge.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)


# --- 2. Model ------------------------------------------------------------
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 -> 32 maps of 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64 maps of 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),                           # one score per digit 0-9
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = DigitNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- 3. Train / evaluate -------------------------------------------------
def train_one_epoch():
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                  # clear old gradients
        loss = loss_fn(model(images), labels)  # forward pass + how wrong we are
        loss.backward()                        # backprop: compute gradients
        optimizer.step()                       # nudge weights downhill
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate():
    model.eval()
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(test_data)


if __name__ == "__main__":
    print(f"Training on {device}, {sum(p.numel() for p in model.parameters()):,} parameters")
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch()
        acc = evaluate()
        print(f"Epoch {epoch}: train loss {avg_loss:.4f} | test accuracy {acc:.2%}")
    print(f"Done in {time.time() - start:.0f}s")
    torch.save(model.state_dict(), "digit_model.pt")
    print("Saved weights to digit_model.pt")
