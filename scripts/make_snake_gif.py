"""Make assets/snake-learning.gif: one game at each stage of training, from random to skilled.

Run:  .venv/bin/python scripts/make_snake_gif.py
"""
import copy
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "snake_rl"))
from snake import SIZE, SnakeGame  # noqa: E402
from train_q import EPSILON_DECAY_EPISODES, EPSILON_END, EPSILON_START, choose_action, q_update  # noqa: E402

STAGES = [0, 300, 1000, 3000]   # snapshot the Q-table after this many training games
MAX_FRAMES = 170                # per stage, so the GIF stays short
BG, INK, AMBER = (14, 15, 18), (232, 228, 219), (242, 169, 59)
DIM, FAINT = (110, 108, 104), (52, 52, 54)
CELL, PAD, TOP = 22, 24, 64
W = H = None


def font(px):
    for path in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, px)
    return ImageFont.load_default()


def snapshots():
    rng, game, q, snaps = random.Random(0), SnakeGame(seed=0), {}, {}
    for episode in range(STAGES[-1] + 1):
        if episode in STAGES:
            snaps[episode] = copy.deepcopy(q)
        epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_EPISODES)
        state, done = game.reset(), False
        while not done:
            action = choose_action(q, state, epsilon, rng)
            nxt, reward, done = game.step(action)
            q_update(q, state, action, reward, nxt, done)
            state = nxt
    return snaps


def frame(game, games_trained, f_small, f_big):
    size = SIZE * CELL + 2 * PAD
    img = Image.new("RGB", (size, size + TOP), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD, 18), f"after {games_trained:,} games", font=f_big, fill=INK)
    label = "random moves" if games_trained == 0 else f"score {game.score}"
    d.text((size - PAD - d.textlength(label, font=f_small), 24), label, font=f_small,
           fill=AMBER if games_trained else DIM)
    c = lambda p: (PAD + (p[0] + 0.5) * CELL, TOP + PAD + (p[1] + 0.5) * CELL)
    for x in range(SIZE):
        for y in range(SIZE):
            cx, cy = c((x, y))
            d.ellipse([cx - 1.2, cy - 1.2, cx + 1.2, cy + 1.2], fill=FAINT)
    pts = [c(p) for p in game.snake]
    if len(pts) > 1:
        d.line(pts, fill=INK, width=int(CELL * 0.36), joint="curve")
    for p in (pts[-1],):
        d.ellipse([p[0] - CELL * .18, p[1] - CELL * .18, p[0] + CELL * .18, p[1] + CELL * .18], fill=INK)
    hx, hy = pts[0]
    d.ellipse([hx - CELL * .32, hy - CELL * .32, hx + CELL * .32, hy + CELL * .32], fill=AMBER)
    fx, fy = c(game.food)
    d.ellipse([fx - CELL * .24, fy - CELL * .24, fx + CELL * .24, fy + CELL * .24], outline=INK, width=2)
    return img


def main():
    f_small, f_big = font(13), font(18)
    frames, durations = [], []
    for stage, q in snapshots().items():
        # Pick a representative game: the median score out of a few seeds.
        results = []
        for seed in range(9):
            g = SnakeGame(seed=seed)
            st, done, rng = g.reset(), False, random.Random(seed)
            while not done:
                st, _, done = g.step(choose_action(q, st, 1.0 if stage == 0 else 0.0, rng))
            results.append((g.score, seed))
        seed = sorted(results)[len(results) // 2][1]
        g = SnakeGame(seed=seed)
        st, done, rng, n = g.reset(), False, random.Random(seed), 0
        while not done and n < MAX_FRAMES:
            frames.append(frame(g, stage, f_small, f_big)); durations.append(70); n += 1
            st, _, done = g.step(choose_action(q, st, 1.0 if stage == 0 else 0.0, rng))
        frames.append(frame(g, stage, f_small, f_big)); durations.append(1100)  # hold on the final frame
    out = ROOT / "assets" / "snake-learning.gif"
    pal = [f.quantize(colors=16, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print(f"{out.relative_to(ROOT)}: {len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
