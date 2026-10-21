"""Train a variational autoencoder (VAE) on MNIST. It learns to squeeze each digit into just
2 numbers and rebuild it from them, without ever seeing a label. Then it can draw new digits.

Run (from ~/Desktop/ai):  .venv/bin/python autoencoder/train_vae.py
Output: autoencoder/vae_model.pt + pictures in autoencoder/samples/
"""
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

HERE = Path(__file__).parent

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LATENT_DIM = 2   # each digit becomes 2 numbers, so the whole "digit space" can be drawn as a 2-D map
BETA = 1.0       # weight of the KL term in vae_loss()

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. Data -------------------------------------------------------------
# Same MNIST files as train.py. No Normalize this time: pixels stay in 0..1 so they can be
# compared directly with the decoder's output. The labels are loaded but never used,
# which is what makes this unsupervised.
transform = transforms.ToTensor()
train_data = datasets.MNIST(str(HERE.parent / "data"), train=True, download=True, transform=transform)
test_data = datasets.MNIST(str(HERE.parent / "data"), train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)


# --- 2. Model ------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(        # 784 pixels -> a compact summary
            nn.Flatten(),
            nn.Linear(28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.to_mu = nn.Linear(256, LATENT_DIM)      # center of this digit's spot in latent space
        self.to_logvar = nn.Linear(256, LATENT_DIM)  # how spread out that spot is (log of the variance)
        self.decoder = nn.Sequential(        # 2 numbers -> 784 pixels
            nn.Linear(LATENT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28 * 28),
        )

    def encode(self, images):
        h = self.encoder(images)
        # Clamp so logvar.exp() can't overflow to inf and turn the whole run into NaN.
        return self.to_mu(h), self.to_logvar(h).clamp(-10, 10)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)  # raw scores (logits); .sigmoid() turns them into pixels

    def forward(self, images):
        mu, logvar = self.encode(images)
        # Reparameterization trick: pick a random point near mu in a way gradients can still flow through.
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z), mu, logvar


def vae_loss(recon_logits, images, mu, logvar):
    """How wrong the VAE is on one batch. Return (total, recon, kl), each summed over the batch.

    Two goals pull against each other:
      recon: rebuild each image accurately.
             F.binary_cross_entropy_with_logits(recon_logits, images, reduction="sum")
      kl:    keep every digit's (mu, logvar) cloud close to a standard normal, so the latent space
             stays smooth and gap-free and random points decode to real-looking digits.
             -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    BETA decides who wins: low BETA gives sharp rebuilds but a lumpy space (bad new digits),
    high BETA gives a smooth space but blurry digits.
    """
    recon = F.binary_cross_entropy_with_logits(recon_logits, images, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + BETA * kl, recon, kl


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- 3. Train / evaluate -------------------------------------------------
def train_one_epoch():
    model.train()
    totals = torch.zeros(3, device=device)
    for images, _ in train_loader:  # "_" = the label we're ignoring
        images = images.to(device)
        recon_logits, mu, logvar = model(images)
        losses = vae_loss(recon_logits, images, mu, logvar)
        optimizer.zero_grad()
        losses[0].backward()
        optimizer.step()
        totals += torch.stack(losses).detach()  # stays on the GPU; one sync per epoch instead of per batch
    return (totals / len(train_data)).tolist()  # per image


@torch.no_grad()
def evaluate():
    model.eval()
    total = 0.0
    for images, _ in test_loader:
        images = images.to(device)
        mu, logvar = model.encode(images)
        # Decode from mu itself (no random sampling) so the score that picks the best epoch isn't luck.
        total += vae_loss(model.decode(mu), images, mu, logvar)[0].item()
    return total / len(test_data)


@torch.no_grad()
def save_samples():
    model.eval()
    out = HERE / "samples"
    out.mkdir(exist_ok=True)

    # 1. Top row: real test digits. Bottom row: what the VAE rebuilt from just 2 numbers each.
    images = torch.stack([test_data[i][0] for i in range(16)]).to(device)
    rebuilt = model.decode(model.encode(images)[0]).sigmoid()
    save_image(torch.cat([images, rebuilt]), out / "reconstructions.png", nrow=16)

    # 2. Brand-new digits: decode random points from the standard normal the KL term aims for.
    save_image(model.decode(torch.randn(64, LATENT_DIM, device=device)).sigmoid(), out / "random.png", nrow=8)
    saved = ["reconstructions.png", "random.png"]

    # 3. A map of the whole latent space: decode a 20x20 grid of points and watch digits morph.
    #    Only possible in 2-D; remove any old map so it can't be mistaken for this run's.
    (out / "latent_map.png").unlink(missing_ok=True)
    if LATENT_DIM == 2:
        grid = torch.linspace(-2.5, 2.5, 20)
        z = torch.cartesian_prod(grid, grid).to(device)
        save_image(model.decode(z).sigmoid(), out / "latent_map.png", nrow=20)
        saved.append("latent_map.png")
    print(f"Saved {', '.join(saved)} to {HERE.name}/samples/")


if __name__ == "__main__":
    print(f"Training on {device}, {sum(p.numel() for p in model.parameters()):,} parameters")
    start = time.time()
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        total, recon, kl = train_one_epoch()
        test_loss = evaluate()
        marker = ""
        if test_loss < best_loss:
            best_loss, best_epoch = test_loss, epoch
            torch.save(model.state_dict(), HERE / "vae_model.pt")
            marker = "  <- best so far, saved"
        print(f"Epoch {epoch}: train loss {total:.1f} (recon {recon:.1f} + kl {kl:.1f}) | test loss {test_loss:.1f}{marker}")
    print(f"Done in {time.time() - start:.0f}s. Saved epoch {best_epoch} to {HERE.name}/vae_model.pt")
    model.load_state_dict(torch.load(HERE / "vae_model.pt", map_location=device, weights_only=True))
    save_samples()
