"""Deep Q-learning (DQN) for Snake: the Q-table from snake_rl/, replaced by a neural network.

Run:  .venv/bin/python snake_dqn/train_dqn.py
Output: snake_dqn/dqn_model.pt + a learning curve printed as it trains.

Same game, same rewards, same 3 actions as snake_rl/. Two things change:
  1. The agent senses more: for each move, how much open space it could still reach afterwards.
     That's a number between 0 and 1, not a yes/no fact, so it can't be a row in a table.
  2. A neural network estimates the Q-values instead of looking them up, so it can handle that.
"""
import random
import sys
import time
from collections import deque
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "snake_rl"))
from snake import DIRS, SIZE, SnakeGame  # noqa: E402  (the exact same game as the Q-table agent)

EPISODES = 1500
GAMMA = 0.95             # discount: how much future reward counts
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
REPLAY_SIZE = 50_000     # how many past moves to remember and learn from
TRAIN_EVERY = 4          # take one gradient step every 4 moves
SYNC_EVERY = 1_000       # copy the online network into the target network this often (in moves)
EPSILON_START, EPSILON_END, EPSILON_DECAY_EPISODES = 1.0, 0.01, 600

N_FEATURES = 14
OUT = HERE / "dqn_model.pt"
torch.set_num_threads(1)  # a tiny network trains faster on one CPU thread than on a GPU


def reachable_fraction(game, start):
    """Flood fill: what fraction of the board could the snake still reach from `start`?
    Its body counts as walls, except the tail, which will have moved out of the way."""
    blocked = set(game.snake[:-1])
    if start in blocked or not (0 <= start[0] < SIZE and 0 <= start[1] < SIZE):
        return 0.0
    seen, stack = {start}, [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in DIRS:
            nxt = (x + dx, y + dy)
            if 0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) / (SIZE * SIZE - len(blocked))


def features(game):
    """The 11 yes/no facts the Q-table agent used, plus open space after each of the 3 moves."""
    hx, hy = game.snake[0]
    space = []
    for turn in (0, 1, -1):  # straight, right, left, matching the action order
        dx, dy = DIRS[(game.dir + turn) % 4]
        space.append(reachable_fraction(game, (hx + dx, hy + dy)))
    return [float(s) for s in game.state()] + space


class QNet(nn.Module):
    """14 numbers about the situation in, 3 Q-values out (one per move)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_FEATURES, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.layers(x)


def choose_action(net, feats, epsilon, rng):
    """Epsilon-greedy, exactly like the Q-table agent, but the values come from the network."""
    if rng.random() < epsilon:
        return rng.randrange(3)
    with torch.no_grad():
        return net(torch.tensor(feats)).argmax().item()


def train_step(net, target, optimizer, replay):
    batch = random.sample(replay, BATCH_SIZE)
    s, a, r, s2, done = (torch.tensor(x) for x in zip(*batch))
    q = net(s).gather(1, a.unsqueeze(1)).squeeze(1)       # Q(state, action taken)
    with torch.no_grad():                                 # the Q-learning target, from the frozen copy
        best_next = target(s2).max(1).values
        goal = r + GAMMA * best_next * (1 - done)
    loss = nn.functional.smooth_l1_loss(q, goal)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train():
    random.seed(0)
    torch.manual_seed(0)
    rng, game = random.Random(0), SnakeGame(seed=0)
    net, target = QNet(), QNet()
    target.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    replay = deque(maxlen=REPLAY_SIZE)
    scores, moves, best_avg = [], 0, -1.0
    start = time.time()
    print(f"{'games':>6} | {'epsilon':>7} | avg score (last 100) | best")
    for episode in range(1, EPISODES + 1):
        epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_EPISODES)
        game.reset()
        feats, done = features(game), False
        while not done:
            action = choose_action(net, feats, epsilon, rng)
            _, reward, done = game.step(action)
            next_feats = features(game)
            replay.append((feats, action, float(reward), next_feats, float(done)))
            feats = next_feats
            moves += 1
            if len(replay) >= BATCH_SIZE and moves % TRAIN_EVERY == 0:
                train_step(net, target, optimizer, replay)
            if moves % SYNC_EVERY == 0:
                target.load_state_dict(net.state_dict())
        scores.append(game.score)
        avg = sum(scores[-100:]) / len(scores[-100:])
        if episode >= 100 and avg > best_avg:  # keep the best network seen, like train.py keeps its best epoch
            best_avg = avg
            torch.save(net.state_dict(), OUT)
        if episode % 100 == 0:
            print(f"{episode:>6} | {epsilon:>7.2f} | {avg:>6.1f} {'#' * int(avg / 2)}".ljust(50) + f"| {max(scores)}")
    print(f"Done in {time.time() - start:.0f}s. Best 100-game average: {best_avg:.1f}. Saved to {OUT.name}")


if __name__ == "__main__":
    train()
