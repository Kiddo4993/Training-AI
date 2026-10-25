"""Watch the trained agent play Snake in the terminal.

Run:  .venv/bin/python snake_rl/play.py            # trained agent
      .venv/bin/python snake_rl/play.py --random   # untrained (random moves), for comparison
      .venv/bin/python snake_rl/play.py --games 100 --fast   # no animation, just average score
"""
import argparse
import json
import random
import time
from pathlib import Path

from snake import ACTIONS, SnakeGame
from train_q import choose_action

parser = argparse.ArgumentParser()
parser.add_argument("--random", action="store_true", help="ignore the Q-table and move randomly")
parser.add_argument("--games", type=int, default=1)
parser.add_argument("--fast", action="store_true", help="skip the animation")
args = parser.parse_args()

q = {} if args.random else json.loads((Path(__file__).parent / "q_table.json").read_text())
epsilon = 1.0 if args.random else 0.0  # fully random vs. always pick the best-known move
rng = random.Random()

scores = []
for g in range(args.games):
    game = SnakeGame()
    state, done = game.reset(), False
    while not done:
        action = choose_action(q, state, epsilon, rng)
        state, _, done = game.step(action)
        if not args.fast:
            print("\033[H\033[J" + game.render())  # clear the terminal, redraw
            print(f"Score: {game.score}   Move: {ACTIONS[action]}")
            time.sleep(0.06)
    scores.append(game.score)

who = "Random agent" if args.random else "Trained agent"
print(f"{who}: average score {sum(scores) / len(scores):.1f} over {len(scores)} game(s), best {max(scores)}")
