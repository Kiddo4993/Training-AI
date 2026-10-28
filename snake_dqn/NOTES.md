# Snake AI, deep Q-learning (DQN)

*Started 2026-10-28 · follow-up to [snake_rl/](../snake_rl/NOTES.md)*

## What it is
The same Snake game, rewards and actions as the Q-table agent, but the **table is replaced by a neural network**. That switch lets the snake sense something a table can't store, and it **more than doubles its score**.

**Training method: deep reinforcement learning (DQN).** It's the method DeepMind used to learn Atari games from pixels in 2013, at a much smaller scale.

## How to run
```bash
cd ~/Desktop/ai
.venv/bin/python snake_dqn/train_dqn.py                      # train (~3 min on a laptop CPU)
.venv/bin/python snake_dqn/play.py                           # watch it play
.venv/bin/python snake_dqn/play.py --games 200 --compare     # DQN vs. Q-table vs. random on the same games
```

## Results (2026-10-28)
**Head-to-head on the same 200 games:**
| Agent | Avg apples | Best |
|---|---|---|
| Random | 0.1 | 1 |
| Q-table ([snake_rl](../snake_rl/NOTES.md)) | 21.4 | 37 |
| **DQN** | **44.3** | **70** |

**Learning curve** (169s, 1,500 games):
| Games | ε | Avg score (last 100) | Best so far |
|---|---|---|---|
| 300 | 0.51 | 1.2 | 5 |
| 500 | 0.18 | 3.3 | 15 |
| 600 | 0.01 | 10.8 | 46 |
| 700 | 0.01 | 28.1 | 55 |
| 1300 | 0.01 | 33.5 | 59 |
| 1500 | 0.01 | 33.0 | 60 |

- The best 100-game training average was **34.4**, and that network is the one saved. The head-to-head score is higher (44.3) because training still made random moves 1% of the time, while evaluation always takes the best move.
- As with the Q-table agent, scores stay near zero while ε is high and jump once it drops. Exploring looks bad until the agent starts using what it learned.

## Why it beats the Q-table
The Q-table snake got stuck around 20 because it only knew about danger **one square** away, so it coiled into its own body. This agent senses 3 extra things: **for each move, the fraction of the board it could still reach afterwards** (computed with a flood fill). Moving into a dead-end pocket now looks bad *before* it's too late.

Why that needs a neural network:
- Those 3 values are **continuous numbers** (for example 0.62, 0.08, 0.95), not yes/no facts. A table needs one row per exact situation, and continuous inputs would make infinitely many rows.
- A network **generalizes**: it learns that "low reachable space = bad" in general, so it handles situations it has never seen exactly.

The new senses do the heavy lifting. Without them, the DQN would face the same blind spot as the table. The neural network is what makes them *possible to use*.

## Key concepts
- **Q-network**: a neural network that takes the state (14 numbers) and outputs 3 Q-values, one per move. It replaces the lookup table.
- **Experience replay** (`REPLAY_SIZE = 50,000`): every move is stored, and training samples random batches of past moves. Learning only from the latest moves would be like studying one page of a textbook over and over. Random batches break up correlations between consecutive moves.
- **Target network** (`SYNC_EVERY = 1,000` moves): a frozen copy of the network computes the "goal" values. Without it, the network chases a target that moves every time it updates, which makes training unstable.
- **Same Q-learning rule, now as a loss:** `loss = Huber( Q(s, a), r + γ · max Q_target(s′) )`. Backpropagation does what `values[action] += α · (target − value)` did in the table version.
- **Huber loss** (`smooth_l1_loss`): like squared error for small mistakes and like absolute error for big ones, so a single surprising crash doesn't produce a huge, destabilizing gradient.
- **Keep the best network:** like `train.py` keeping its best epoch, the score varies during training, so the best 100-game average is what gets saved.

## Experiments to try next
- [ ] **Remove the flood-fill features** (use only the 11 booleans, and set `N_FEATURES = 11`). Does DQN do any better than the table? This tests whether the network or the senses made the difference.
- [ ] **No target network:** set `SYNC_EVERY = 1`. Is training less stable?
- [ ] **Tiny replay memory:** set `REPLAY_SIZE = 500`. What happens?
- [ ] **Bigger board** (`SIZE = 20` in `snake_rl/snake.py`). The flood-fill features are fractions, so they still make sense. Does the trained network transfer?
- [ ] **Raw pixels:** feed the whole 12×12 grid as input with a small CNN instead of hand-made features. That's the original Atari DQN idea. It's much harder, but nothing is hand-designed.

## My notes
<!-- add your own observations here -->
