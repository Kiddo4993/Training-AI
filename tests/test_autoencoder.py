"""Variational autoencoder (autoencoder/)."""
from pathlib import Path

import torch

from train_vae import LATENT_DIM, VAE, device, test_loader, vae_loss

CHECKPOINT = Path(__file__).resolve().parent.parent / "autoencoder" / "vae_model.pt"


def test_vae_squeezes_digits_to_2_numbers_and_back():
    recon, mu, logvar = VAE()(torch.rand(3, 1, 28, 28))
    assert recon.shape == (3, 1, 28, 28) and mu.shape == logvar.shape == (3, LATENT_DIM)


@torch.no_grad()
def test_trained_vae_rebuilds_test_digits():
    model = VAE().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    torch.manual_seed(0)
    images, _ = next(iter(test_loader))
    images = images.to(device)
    recon, mu, logvar = model(images)
    total, _, _ = vae_loss(recon, images, mu, logvar)
    assert total.item() / len(images) < 160  # training reached ~143 per image
