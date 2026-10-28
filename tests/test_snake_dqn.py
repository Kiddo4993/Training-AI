"""Deep Q-learning Snake agent (snake_dqn/)."""
import random
from pathlib import Path

import torch

from snake import SnakeGame
from train_dqn import N_FEATURES, OUT, QNet, choose_action, features, reachable_fraction


def test_features_add_open_space_to_the_11_facts():
    feats = features(SnakeGame(seed=0))
    assert len(feats) == N_FEATURES == 14
    assert all(0.0 <= f <= 1.0 for f in feats)


def test_flood_fill_sees_a_trap():
    game = SnakeGame(seed=0)
    game.reset()
    game.snake = [(1, 1), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]  # head at (1,1), walled in
    assert reachable_fraction(game, (2, 1)) > 0.5          # the way out
    assert reachable_fraction(game, (-1, 1)) == 0.0        # off the board
    assert reachable_fraction(game, (0, 1)) == 0.0         # into its own body


def test_q_network_outputs_one_value_per_move():
    assert QNet()(torch.zeros(5, N_FEATURES)).shape == (5, 3)


def test_trained_dqn_beats_the_q_table_plateau():
    net = QNet()
    net.load_state_dict(torch.load(OUT, map_location="cpu", weights_only=True))
    net.eval()
    scores = []
    for i in range(20):
        game, rng = SnakeGame(seed=1000 + i), random.Random(i)
        game.reset()
        done = False
        while not done:
            _, _, done = game.step(choose_action(net, features(game), 0.0, rng))
        scores.append(game.score)
    assert sum(scores) / len(scores) >= 30  # the Q-table agent averages ~21
