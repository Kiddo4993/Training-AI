# Snake AI (Reinforcement Learning)

*Started 2026-10-25*

## What it is
An agent that learns to play Snake **by playing it**, with no examples and nobody showing it good moves. It starts out moving randomly, gets **+10** for eating an apple and **−10** for dying, and over 3,000 games works out a strategy on its own.

**Training method: reinforcement learning (tabular Q-learning).** This is the 5th method in this repo, and the only one that doesn't learn from a dataset. See Section 3 of [AI Training Methods](<../AI Training Methods/TRAINING_METHODS.md>).

| Project | Learns from |
|---|---|
| Digit recognizer | Labeled examples (supervised) |
| Handwriting fine-tune | A pretrained model + a few new examples (transfer) |
| Tiny GPT | Text that labels itself (self-supervised) |
| VAE | Unlabeled images (unsupervised) |
| **Snake** | **Rewards from its own actions (reinforcement)** |

## How to run
```bash
cd ~/Desktop/ai
.venv/bin/python snake_rl/train_q.py                       # train (~2 seconds)
.venv/bin/python snake_rl/play.py                          # watch it play
.venv/bin/python snake_rl/play.py --random                 # watch an untrained snake, for comparison
.venv/bin/python snake_rl/play.py --games 200 --fast       # average score over 200 games
```

## Results (2026-10-25)
| Games played | Epsilon (how random) | Avg score (last 100 games) | Best so far |
|---|---|---|---|
| 250 | 0.83 | 0.4 | 2 |
| 500 | 0.67 | 0.9 | 3 |
| 1000 | 0.34 | 2.5 | 9 |
| 1250 | 0.18 | 3.9 | 11 |
| **1500** | **0.01** | **11.7** | **31** |
| 2000 | 0.01 | 18.2 | 41 |
| 3000 | 0.01 | 17.0 | 41 |

- **Trained for 2 seconds.** Over 200 test games, the trained agent averaged **20.1** (best 39) and the random agent averaged **0.2** (best 2).
- **The big jump at game 1,500** is when epsilon reached its minimum. Before that, the agent was often moving randomly *on purpose* to explore, so its scores looked bad even though it was learning.
- **It leveled off at about 18–20.** It never gets much better, for the reason below.
- It only ever saw **256 of the 2,048 possible states**. Many combinations can't happen (for example, "heading up" and "heading left" at the same time).

### Why it gets stuck at about 20
The snake can only sense **one square** around its head. As it grows, it coils into pockets of its own body and traps itself:
```
. . . o o o o o o . . .
. . . o o . . o o . . .
. . o o o . . o . . . .
. . o @ o . . o . . . .     @ = head, boxed in by its own body
* . o o o o o o . . . .     * = the apple it was heading for
```
By the time "danger ahead/left/right" are all true, it's too late. **The agent can only be as smart as what it can sense.** To score higher, it needs a richer state (see the experiments below).

## Key concepts
- **Agent**: the learner (the snake's brain)
- **Environment**: the world it acts in (`snake.py`)
- **State**: what the agent senses each step (11 yes/no facts here)
- **Action**: what it can do (go straight, turn right, turn left)
- **Reward**: the score signal (+10 apple, −10 death, 0 otherwise)
- **Episode**: one full game
- **Q-value, Q(state, action)**: "how good is this action in this situation, counting future rewards too?"
- **Q-table**: the agent's whole brain, a lookup table with 3 numbers per state (`q_table.json`, which you can open and read)

### The Q-learning rule (`q_update` in `train_q.py`)
```
Q(state, action) ← Q + α × [ reward + γ × max Q(next_state) − Q ]
```
- **α (alpha) = 0.1, the learning rate:** how far each update moves the value
- **γ (gamma) = 0.9, the discount:** future rewards count, but a bit less than immediate ones. An apple 10 steps away is worth 0.9¹⁰ ≈ 35% of an apple now
- The `max Q(next_state)` part is the key idea. Value **flows backwards**: the move right before eating gets credit, then the move before that, and so on. Eventually "turn toward the food" scores well even from far away

### Exploration vs. exploitation (`choose_action`)
- **Explore:** try a random move, which might discover something better
- **Exploit:** do the best move you know so far
- **Epsilon-greedy:** explore with probability ε, otherwise exploit. ε starts at 1.0 (all random) and decays to 0.01 by game 1,500
- With too little exploration it never finds good strategies. With too much it never uses what it learned

### Tabular vs. deep reinforcement learning
This works because there are only 2,048 states, so a table can list them all. Chess, Go or a video game from raw pixels have far too many states for a table, so **deep Q-learning (DQN)** replaces the table with a neural network that *estimates* Q-values. That's how DeepMind's agent learned Atari games.

## Experiments to try next
- [ ] **Reward shaping:** set `REWARD_STEP = -0.1` in `snake.py` (a small penalty every step). Does it chase apples more directly? Does it die more?
- [ ] **No exploration:** set `EPSILON_START = 0.0`. Does it ever learn?
- [ ] **Short-sighted agent:** set `GAMMA = 0.0` so only immediate rewards count. What happens?
- [ ] **Richer state:** in `state()`, add facts like "is my tail reachable?" or "danger 2 squares ahead". Can it get past 20?
- [ ] **Bigger board:** set `SIZE = 20`. Does the same Q-table still work?
- [ ] **Deep Q-learning:** replace the table with a small PyTorch network (reusing what you learned in the digit project)

## My notes
<!-- add your own observations here -->
