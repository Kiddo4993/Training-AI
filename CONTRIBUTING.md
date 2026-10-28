# Contributing

Thanks for helping! This repo exists to teach, so the bar for a contribution is simple: **does it help someone understand how AI is trained?**

## Good first contributions
- **Run an experiment.** Every project's `NOTES.md` ends with an "Experiments to try next" list. Try one and open an issue (or PR) with what happened, even if it failed. Failed experiments teach a lot.
- **Fix confusing explanations.** If a comment or note confused you, it will confuse the next person too.
- **Add a test** for behaviour that isn't covered yet.
- **Make it run somewhere new**: Windows, Linux, CUDA GPUs, Google Colab.

## Bigger contributions
New projects are welcome if they teach a training method or idea the repo doesn't cover yet (for example DQN, PPO, diffusion, contrastive learning). Open an issue first so we can agree on the scope. Each project should:
1. Live in its own folder with a `NOTES.md` (what it is, how to run, key concepts, results table, experiments)
2. Train in **under ~10 minutes on a laptop**
3. Use plain PyTorch (or plain Python), with no high-level training frameworks hiding the loop
4. Come with tests in `tests/` that check the trained result

## Development setup
```bash
git clone https://github.com/Kiddo4993/train-ai-from-scratch.git
cd train-ai-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

## Style
- **Readability beats cleverness.** Code is written for beginners to read.
- Comments explain *why* something is done, not what the line says.
- Keep hyperparameters as named constants at the top of each script.
- If your change affects results, update that project's results table.

By contributing, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md) and that your contributions are licensed under the [MIT License](LICENSE).
