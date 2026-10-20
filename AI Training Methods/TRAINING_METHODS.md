# AI Training Methods

A reference guide to the main ways AI models are trained, grouped by category. The digit recognizer in this repo uses **supervised learning** (`train.py`) and then **transfer learning / fine-tuning** (`finetune.py`).

---

## 1. Learning paradigms (where the training signal comes from)

| Method | How it learns | Example uses |
|---|---|---|
| **Supervised learning** | From labeled pairs (input → correct output) | Image classification (this repo's MNIST model), spam detection |
| **Unsupervised learning** | Finds structure in unlabeled data | Clustering (k-means), PCA, anomaly detection |
| **Self-supervised learning** | Creates its own labels from the data | LLM next-token prediction, BERT masked words, contrastive learning (SimCLR, CLIP) |
| **Semi-supervised learning** | A few labels plus lots of unlabeled data | Pseudo-labeling, consistency training |
| **Reinforcement learning (RL)** | Trial and error, guided by a reward signal | Game AI (AlphaGo), robotics, game NPC agents |
| **Imitation learning** | Copies expert demonstrations | Behavior cloning, self-driving |
| **Transfer learning** | Reuses a pretrained model for a new task | Fine-tuning ResNet or BERT on your own data (this repo's `finetune.py`) |
| **Meta-learning** | Learns how to learn new tasks quickly | MAML, few-shot learning |
| **Online / continual learning** | Keeps learning from a data stream without forgetting | Recommendation systems |
| **Federated learning** | Trains on many devices without collecting the raw data centrally | Phone keyboard prediction |
| **Active learning** | Model picks which examples a human should label next | Cutting labeling costs |

## 2. Optimization (how the weights get updated)

- **Gradient descent family:** batch GD, stochastic GD (SGD), mini-batch GD
- **Momentum methods:** Momentum, Nesterov
- **Adaptive optimizers:** AdaGrad, RMSProp, **Adam / AdamW** (the usual default), Lion, Adafactor, Muon
- **Backpropagation:** the chain-rule algorithm that computes gradients for neural nets
- **Gradient-free methods:** genetic/evolutionary algorithms, NEAT, CMA-ES, random search, simulated annealing
- **Classical ML solvers:** closed-form least squares, SMO (for SVMs), greedy tree splitting (decision trees, XGBoost)

## 3. Reinforcement learning methods

- **Value-based:** Q-learning, Deep Q-Networks (DQN), Double DQN
- **Policy gradient:** REINFORCE, **PPO**, TRPO
- **Actor-critic:** A2C/A3C, SAC, DDPG, TD3
- **Model-based:** MuZero, Dreamer, Monte Carlo Tree Search with learned models
- **Multi-agent RL and self-play:** AlphaZero, OpenAI Five

## 4. How large language models are trained

1. **Pretraining:** next-token prediction on trillions of tokens
2. **Supervised fine-tuning (SFT) / instruction tuning:** training on example prompt → good response pairs
3. **Preference alignment:**
   - **RLHF:** humans rank outputs, a reward model learns those preferences, then PPO optimizes against it
   - **DPO / IPO / KTO:** optimize directly on preference pairs, with no separate reward model
   - **RLAIF / Constitutional AI:** an AI gives the feedback, guided by a written set of principles
4. **RL with verifiable rewards (RLVR):** rewards correct math answers or passing code tests (GRPO, for example). This is how reasoning models are trained.
5. **Distillation:** a small "student" model learns to imitate a large "teacher"
6. **Parameter-efficient fine-tuning (PEFT):** **LoRA**, QLoRA, adapters, prefix/prompt tuning. These train under 1% of the weights.

## 5. Regularization (preventing overfitting)

Dropout, weight decay (L2), L1, early stopping, data augmentation, label smoothing, batch/layer normalization, mixup/cutmix, noise injection.

## 6. Training strategies and tricks

- **Learning-rate schedules:** warmup, cosine decay, step decay, one-cycle
- **Curriculum learning:** start with easy examples, then move to harder ones
- **Mixed precision (FP16/BF16) and gradient accumulation/checkpointing:** train bigger models on less memory
- **Distributed training:** data parallelism, model/tensor/pipeline parallelism, FSDP/ZeRO
- **Hyperparameter search:** grid, random, Bayesian optimization, population-based training
- **Ensembling:** bagging (random forests), boosting (AdaBoost, XGBoost), stacking
- **Adversarial training:** GANs (generator vs. discriminator), training on adversarial examples for robustness
- **Diffusion training:** learn to remove noise step by step (Stable Diffusion)

---

## The core training loop

Almost every deep learning method above is a variation on this loop. They mostly change *where the loss comes from* (labels, rewards, preferences, the data itself) or *how the update step is taken*.

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        pred = model(x)              # forward pass
        loss = loss_fn(pred, y)      # how wrong is it?
        optimizer.zero_grad()
        loss.backward()              # backpropagation
        optimizer.step()             # update weights (e.g. AdamW)
```

Real systems stack several methods. An LLM is self-supervised in pretraining, supervised in SFT, trained with RL in alignment, and often LoRA-fine-tuned afterward.
