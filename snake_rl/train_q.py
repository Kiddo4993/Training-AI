"""Teach an agent to play Snake with tabular Q-learning (reinforcement learning).

Run:  .venv/bin/python snake_rl/train_q.py
Output: snake_rl/q_table.json + a learning curve printed as it trains.
"""
import json
import random
import time
from pathlib import Path

from snake import ACTIONS, SnakeGame

EPISODES = 3000       # games to play while learning
ALPHA = 0.1           # learning rate: how far each update moves a Q-value
GAMMA = 0.9           # discount: how much future reward counts vs. reward right now
EPSILON_START = 1.0   # start fully random (explore)...
EPSILON_END = 0.01    # ...end almost always greedy (exploit)
EPSILON_DECAY_EPISODES = 1500

OUT = Path(__file__).parent / "q_table.json"


def key(state):
    """Turn a state of 11 booleans into a short string like '01001000101'."""
    return "".join("1" if s else "0" for s in state)


def choose_action(q, state, epsilon, rng):
    """Epsilon-greedy: explore with probability epsilon, otherwise take the best-known action."""
    if rng.random() < epsilon:
        return rng.randrange(len(ACTIONS))
    values = q.get(key(state), [0.0] * len(ACTIONS))
    return values.index(max(values))


def q_update(q, state, action, reward, next_state, done):
    """The Q-learning rule. Nudge Q(state, action) toward:
    reward now + GAMMA * (best value available from the next state)."""
    values = q.setdefault(key(state), [0.0] * len(ACTIONS))
    future = 0.0 if done else max(q.get(key(next_state), [0.0] * len(ACTIONS)))
    target = reward + GAMMA * future
    values[action] += ALPHA * (target - values[action])


def train():
    rng = random.Random(0)
    game = SnakeGame(seed=0)
    q = {}
    scores = []
    start = time.time()
    print(f"{'games':>6} | {'epsilon':>7} | avg score (last 100) | best")
    for episode in range(1, EPISODES + 1):
        epsilon = max(EPSILON_END,
                      EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_EPISODES)
        state, done = game.reset(), False
        while not done:
            action = choose_action(q, state, epsilon, rng)
            next_state, reward, done = game.step(action)
            q_update(q, state, action, reward, next_state, done)
            state = next_state
        scores.append(game.score)
        if episode % 250 == 0:
            avg = sum(scores[-100:]) / 100
            print(f"{episode:>6} | {epsilon:>7.2f} | {avg:>6.1f} {'#' * int(avg)}".ljust(50)
                  + f"| {max(scores)}")
    print(f"Done in {time.time() - start:.0f}s. Learned values for {len(q)} of 2048 possible states.")
    OUT.write_text(json.dumps(q, indent=0))
    print(f"Saved Q-table to {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}")


if __name__ == "__main__":
    train()
