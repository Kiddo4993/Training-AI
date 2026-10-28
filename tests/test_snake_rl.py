"""Snake environment + Q-learning agent (snake_rl/)."""
import json
import random
from pathlib import Path

import pytest

from snake import SIZE, SnakeGame
from train_q import ACTIONS, choose_action, q_update

Q_TABLE = Path(__file__).resolve().parent.parent / "snake_rl" / "q_table.json"


def play(q, epsilon, games, seed):
    rng, scores = random.Random(seed), []
    for g in range(games):
        game = SnakeGame(seed=seed + g)
        state, done = game.reset(), False
        while not done:
            state, _, done = game.step(choose_action(q, state, epsilon, rng))
        scores.append(game.score)
    return sum(scores) / len(scores)


def test_state_has_11_booleans():
    state = SnakeGame(seed=0).reset()
    assert len(state) == 11 and all(isinstance(s, bool) for s in state)


def test_hitting_a_wall_ends_the_game_with_a_penalty():
    game = SnakeGame(seed=0)
    game.reset()
    done = False
    for _ in range(SIZE):  # keep going straight (right) until the wall
        _, reward, done = game.step(0)
        if done:
            break
    assert done and reward < 0


def test_q_update_moves_value_toward_the_reward():
    q, state = {}, SnakeGame(seed=0).reset()
    q_update(q, state, action=0, reward=10, next_state=state, done=True)
    values = next(iter(q.values()))
    assert values[0] > 0 and values[1] == values[2] == 0


def test_short_training_run_learns_something():
    rng, game, q = random.Random(0), SnakeGame(seed=0), {}
    for episode in range(1200):
        epsilon = max(0.01, 1 - episode / 600)
        state, done = game.reset(), False
        while not done:
            action = choose_action(q, state, epsilon, rng)
            next_state, reward, done = game.step(action)
            q_update(q, state, action, reward, next_state, done)
            state = next_state
    assert play(q, 0.0, games=30, seed=100) > 3 * max(play({}, 1.0, games=30, seed=100), 0.5)


@pytest.mark.skipif(not Q_TABLE.exists(), reason="no trained q_table.json")
def test_trained_agent_beats_random_by_a_lot():
    q = json.loads(Q_TABLE.read_text())
    assert len(ACTIONS) == 3
    assert play(q, 0.0, games=50, seed=0) >= 10
    assert play({}, 1.0, games=50, seed=0) < 1
