"""Watch the DQN agent play Snake, or compare it with the Q-table agent.

Run:  .venv/bin/python snake_dqn/play.py                         # watch the DQN agent
      .venv/bin/python snake_dqn/play.py --games 200 --fast      # average score, no animation
      .venv/bin/python snake_dqn/play.py --games 200 --compare   # DQN vs. Q-table vs. random, same games
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from train_dqn import HERE, OUT, QNet, choose_action, features

sys.path.insert(0, str(HERE.parent / "snake_rl"))
from snake import ACTIONS, SnakeGame  # noqa: E402
import train_q  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--games", type=int, default=1)
parser.add_argument("--fast", action="store_true", help="skip the animation")
parser.add_argument("--compare", action="store_true", help="also run the Q-table and random agents")
args = parser.parse_args()

net = QNet()
net.load_state_dict(torch.load(OUT, map_location="cpu", weights_only=True))
net.eval()
q_table = json.loads((HERE.parent / "snake_rl" / "q_table.json").read_text())

agents = {"DQN": lambda g, rng: choose_action(net, features(g), 0.0, rng)}
if args.compare:
    agents["Q-table"] = lambda g, rng: train_q.choose_action(q_table, g.state(), 0.0, rng)
    agents["Random"] = lambda g, rng: rng.randrange(3)

animate = not (args.fast or args.compare)
for name, act in agents.items():
    scores = []
    for i in range(args.games):
        game, rng = SnakeGame(seed=1000 + i), random.Random(i)  # same games for every agent
        game.reset()
        done = False
        while not done:
            action = act(game, rng)
            _, _, done = game.step(action)
            if animate:
                print("\033[H\033[J" + game.render())
                print(f"Score: {game.score}   Move: {ACTIONS[action]}")
                time.sleep(0.05)
        scores.append(game.score)
    print(f"{name:>8}: average score {sum(scores) / len(scores):.1f} over {len(scores)} game(s), best {max(scores)}")
