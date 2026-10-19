"""Load the trained model and show it guessing a few random test digits (drawn in ASCII).

Run:  .venv/bin/python predict.py
"""
import random

import torch

from train import DigitNet, test_data

model = DigitNet()
model.load_state_dict(torch.load("digit_model.pt", map_location="cpu", weights_only=True))
model.eval()

for i in random.sample(range(len(test_data)), 5):
    image, label = test_data[i]
    with torch.no_grad():
        probs = model(image.unsqueeze(0)).softmax(dim=1)[0]
    guess = probs.argmax().item()

    # Draw the 28x28 image, skipping every other row so it isn't stretched.
    for row in image[0][::2]:
        print("".join("#" if v > 0.5 else "." if v > -0.2 else " " for v in row))
    mark = "correct" if guess == label else "WRONG"
    print(f"Actual: {label}  Guess: {guess}  ({probs[guess]:.1%} confident) -> {mark}\n")
