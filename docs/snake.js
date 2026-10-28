// Snake from snake_rl/snake.py, ported to JavaScript with identical rules, state and actions,
// so the Q-table trained in Python plays exactly the same way here.

export const SIZE = 12;
const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // up, right, down, left (clockwise)
const TURN = [0, 1, -1];                         // actions: straight, right, left

export class SnakeGame {
  constructor(random = Math.random) {
    this.random = random;
    this.reset();
  }

  reset() {
    const mid = Math.floor(SIZE / 2);
    this.snake = [[mid, mid], [mid - 1, mid], [mid - 2, mid]]; // head first
    this.dir = 1;
    this.score = 0;
    this.stepsSinceFood = 0;
    this.placeFood();
    return this.state();
  }

  occupied(x, y, upTo = this.snake.length) {
    for (let i = 0; i < upTo; i++) if (this.snake[i][0] === x && this.snake[i][1] === y) return true;
    return false;
  }

  placeFood() {
    const free = [];
    for (let x = 0; x < SIZE; x++) for (let y = 0; y < SIZE; y++) if (!this.occupied(x, y)) free.push([x, y]);
    this.food = free[Math.floor(this.random() * free.length)];
  }

  hits(x, y) {
    return x < 0 || x >= SIZE || y < 0 || y >= SIZE || this.occupied(x, y, this.snake.length - 1);
  }

  /** 11 yes/no facts, in the same order as snake.py, encoded as a key like "01001000101". */
  state() {
    const [hx, hy] = this.snake[0];
    const look = (t) => DIRS[(this.dir + t + 4) % 4];
    const [a, r, l] = [look(0), look(1), look(-1)];
    const [fx, fy] = this.food;
    return [
      this.hits(hx + a[0], hy + a[1]), this.hits(hx + r[0], hy + r[1]), this.hits(hx + l[0], hy + l[1]),
      this.dir === 0, this.dir === 1, this.dir === 2, this.dir === 3,
      fy < hy, fx > hx, fy > hy, fx < hx,
    ].map((b) => (b ? "1" : "0")).join("");
  }

  /** Returns { done }. */
  step(action) {
    this.dir = (this.dir + TURN[action] + 4) % 4;
    const [dx, dy] = DIRS[this.dir];
    const head = [this.snake[0][0] + dx, this.snake[0][1] + dy];
    this.stepsSinceFood++;
    if (this.hits(head[0], head[1]) || this.stepsSinceFood > 100 * this.snake.length) return { done: true };
    this.snake.unshift(head);
    if (head[0] === this.food[0] && head[1] === this.food[1]) {
      this.score++;
      this.stepsSinceFood = 0;
      if (this.snake.length === SIZE * SIZE) return { done: true };
      this.placeFood();
    } else {
      this.snake.pop();
    }
    return { done: false };
  }
}

/** Trained: take the best-known action from the Q-table. Untrained (qTable null): move randomly. */
export function chooseAction(qTable, stateKey, random = Math.random) {
  if (!qTable) return Math.floor(random() * 3);
  const values = qTable[stateKey] || [0, 0, 0];
  return values.indexOf(Math.max(...values));
}
