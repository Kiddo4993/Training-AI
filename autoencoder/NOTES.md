# Variational Autoencoder (VAE) on MNIST

*Started 2026-10-20*

## What it is
Two networks back to back:
- **Encoder**: squeezes a 28×28 digit (784 pixels) down to just **2 numbers**
- **Decoder**: rebuilds the full image from those 2 numbers

Because only 2 numbers get through the middle, the VAE has to discover on its own what makes a 7 different from a 1. It never sees a label.

**Training method: unsupervised and generative.** Once trained, you can feed the decoder *any* 2 numbers and it draws a digit that never existed.

## How to run
```bash
cd ~/Desktop/ai
.venv/bin/python autoencoder/train_vae.py   # ~1 min, needs vae_loss()
```
Then look in `autoencoder/samples/`:
| File | What it shows |
|---|---|
| `reconstructions.png` | Top: real digits. Bottom: rebuilt from 2 numbers each |
| `random.png` | 64 brand-new digits from random points |
| `latent_map.png` | The whole 2-D "digit space" as a 20×20 grid, with digits morphing into each other |

## Key concepts
- **Latent space**: the space of those 2 numbers. Similar digits end up close together.
- **Reconstruction loss**: how different the rebuilt image is from the original.
- **KL divergence**: penalizes the encoder for spreading digits far apart or leaving gaps, so every point in the space decodes to something digit-like.
- **BETA**: balances the two losses. This is the main design choice in this project.
- **Reparameterization trick**: sampling is random, which normally blocks gradients. Writing it as `mu + noise × std` lets backprop flow through.
- **Autoencoder vs. VAE**: a plain autoencoder only learns to compress. The KL term is what makes a VAE able to *generate*.

## Results
| Run | BETA | Epochs | Best test loss | Notes |
|---|---|---|---|---|
| 1 (2026-10-21) | 1.0 | 10 | 143.4 (recon 137 + kl 6.4) | 42s. Test loss improved every epoch, so more epochs would still help. Rebuilds are good, but with only 2 numbers it mixes up look-alikes (4↔9, 5↔3) |
