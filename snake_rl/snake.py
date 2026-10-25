"""Snake, written as a reinforcement learning *environment*.

The agent never sees the board. Each step it gets a small summary (the state),
picks one of 3 actions, and the game hands back a reward plus the next state.
"""
import random

SIZE = 12  # board is SIZE x SIZE

# Directions in clockwise order, so "turn right" = index + 1, "turn left" = index - 1.
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
ACTIONS = ["straight", "right", "left"]

REWARD_FOOD = 10
REWARD_DEATH = -10
REWARD_STEP = 0


class SnakeGame:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        mid = SIZE // 2
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]  # head first
        self.dir = 1  # facing right
        self.score = 0
        self.steps_since_food = 0
        self._place_food()
        return self.state()

    def _place_food(self):
        free = [(x, y) for x in range(SIZE) for y in range(SIZE) if (x, y) not in self.snake]
        self.food = self.rng.choice(free)

    def _hits(self, point):
        x, y = point
        return not (0 <= x < SIZE and 0 <= y < SIZE) or point in self.snake[:-1]

    def state(self):
        """11 yes/no facts about the situation. 2^11 = 2048 possible states."""
        hx, hy = self.snake[0]
        ahead, right, left = (DIRS[(self.dir + t) % 4] for t in (0, 1, -1))
        fx, fy = self.food
        return (
            self._hits((hx + ahead[0], hy + ahead[1])),  # danger straight ahead
            self._hits((hx + right[0], hy + right[1])),  # danger to the right
            self._hits((hx + left[0], hy + left[1])),    # danger to the left
            self.dir == 0, self.dir == 1, self.dir == 2, self.dir == 3,  # heading
            fy < hy, fx > hx, fy > hy, fx < hx,  # food is up / right / down / left
        )

    def step(self, action):
        """Apply an action (0 straight, 1 right, 2 left). Returns (state, reward, done)."""
        self.dir = (self.dir + (0, 1, -1)[action]) % 4
        dx, dy = DIRS[self.dir]
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        self.steps_since_food += 1

        # Die on walls/self, or if it wanders too long without eating (stops endless loops).
        if self._hits(head) or self.steps_since_food > 100 * len(self.snake):
            return self.state(), REWARD_DEATH, True

        self.snake.insert(0, head)
        if head == self.food:
            self.score += 1
            self.steps_since_food = 0
            if len(self.snake) == SIZE * SIZE:
                return self.state(), REWARD_FOOD, True  # filled the board
            self._place_food()
            return self.state(), REWARD_FOOD, False
        self.snake.pop()
        return self.state(), REWARD_STEP, False

    def render(self):
        rows = []
        for y in range(SIZE):
            row = ""
            for x in range(SIZE):
                if (x, y) == self.snake[0]:
                    row += "@ "
                elif (x, y) in self.snake:
                    row += "o "
                elif (x, y) == self.food:
                    row += "* "
                else:
                    row += ". "
            rows.append("|" + row.rstrip() + "|")
        border = "+" + "-" * (SIZE * 2 - 1) + "+"
        return "\n".join([border, *rows, border])
