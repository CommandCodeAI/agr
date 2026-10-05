"""Snake as a decision task for Agr: the board, the text the model reads, the
questions it is asked, and the policy that turns its answers into a move.

The model is asked five things per tick in one call: which direction to move,
and whether each of the four directions is safe. The move is taken as the
model's most probable direction among the legal ones (walls and the body are
the environment's rules, not something to gamble on), the safety answers are
reported against the truth, and a loop breaker steers toward the food when the
snake starts cycling through the same cells.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

N = 12
DIRS: dict[str, tuple[int, int]] = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}


def questions(detail: str = "full") -> dict:
    """`detail="compact"` trades wording for tokens: a decision costs one prefill of the
    state plus the question branches, and that prefill is compute-bound, so halving the
    tokens halves the time."""
    if detail == "compact":
        q = {"move": {"type": "choice", "instructions": "Best direction to move?", "criteria": {"up": "", "down": "", "left": "", "right": ""}}}
        for d in DIRS:
            q[f"safe_{d}"] = {"type": "boolean", "instructions": f"Is moving {d} safe?"}
        return q
    q = {
        "move": {
            "type": "choice",
            "instructions": "Which direction should the snake move next? Pick a direction whose adjacent cell is empty or food, never a wall or a body cell, and prefer the direction that brings the head closer to the food.",
            "criteria": {"up": "", "down": "", "left": "", "right": ""},
        }
    }
    for d in DIRS:
        q[f"safe_{d}"] = {"type": "boolean", "instructions": f"Is it safe for the snake to move {d} on the next step, that is, the adjacent cell in that direction is inside the grid and not part of the snake's body?"}
    return q


@dataclass
class Game:
    rng: random.Random
    snake: list[tuple[int, int]] = field(default_factory=lambda: [(5, 6), (4, 6), (3, 6)])
    dir: str = "right"
    score: int = 0
    steps: int = 0
    food: tuple[int, int] = (0, 0)
    history: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.place_food()

    def place_food(self) -> None:
        free = [(x, y) for x in range(N) for y in range(N) if (x, y) not in self.snake]
        self.food = self.rng.choice(free) if free else (-1, -1)  # a full board has nowhere left: no food

    def cell(self, x: int, y: int) -> str:
        if not (0 <= x < N and 0 <= y < N):
            return "wall"
        if (x, y) == self.food:
            return "food"
        if (x, y) in self.snake:
            return "body"
        return "empty"

    def ahead(self, d: str) -> tuple[int, int]:
        dx, dy = DIRS[d]
        hx, hy = self.snake[0]
        return hx + dx, hy + dy

    def truth_safe(self, d: str) -> bool:
        if d == OPP[self.dir]:
            return False
        nx, ny = self.ahead(d)
        what = self.cell(nx, ny)
        if what == "body" and (nx, ny) == self.snake[-1] and (nx, ny) != self.food:
            return True  # the tail moves out of the way this step
        return what in ("empty", "food")

    def legal(self) -> list[str]:
        return [d for d in DIRS if self.truth_safe(d)]

    def distance(self, d: str) -> int:
        nx, ny = self.ahead(d)
        return abs(nx - self.food[0]) + abs(ny - self.food[1])

    def state(self, detail: str = "full") -> str:
        if detail == "compact":
            return self.compact_state()
        hx, hy = self.snake[0]
        fx, fy = self.food
        dxf, dyf = fx - hx, fy - hy
        toward = [d for d, ok in (("right", dxf > 0), ("left", dxf < 0), ("down", dyf > 0), ("up", dyf < 0)) if ok]
        adjacent = []
        for d in DIRS:
            nx, ny = self.ahead(d)
            what = self.cell(nx, ny)
            if d == OPP[self.dir]:
                what = "your own neck (reversing is not allowed)"
            adjacent.append(f"{d} ({nx},{ny}): {what}")
        grid = []
        for y in range(N):
            grid.append("".join("H" if (x, y) == (hx, hy) else "F" if (x, y) == self.food else "o" if (x, y) in self.snake else "." for x in range(N)))
        body = ", ".join(f"({x},{y})" for x, y in self.snake[1:]) or "none"
        return (
            f"Snake game on a {N}x{N} grid. Coordinates are (column,row); (0,0) is the top-left cell. up decreases the row, down increases it, left decreases the column, right increases it. Moving into a wall or into the snake's own body ends the game.\n"
            f"Head at ({hx},{hy}), currently moving {self.dir}. Body cells behind the head: {body}.\n"
            f"Adjacent cells: " + "; ".join(adjacent) + ".\n"
            f"Food at ({fx},{fy}), which is {abs(dxf)} column(s) and {abs(dyf)} row(s) away; the directions that bring the head closer are: {', '.join(toward) or 'none, the head is on the food'}.\n"
            f"The head has just come from these cells and should not double back onto them: {', '.join(f'({x},{y})' for x, y in self.recent()[-6:]) or 'none'}.\n"
            f"Grid (H = head, o = body, F = food, . = empty), rows 0 to {N - 1}:\n" + "\n".join(grid)
        )

    def compact_state(self) -> str:
        """The same facts without the drawn grid or the rules paragraph: roughly a
        quarter of the tokens, which is roughly a quarter of the decision time."""
        hx, hy = self.snake[0]
        fx, fy = self.food
        dxf, dyf = fx - hx, fy - hy
        toward = ", ".join(d for d, ok in (("right", dxf > 0), ("left", dxf < 0), ("down", dyf > 0), ("up", dyf < 0)) if ok) or "none"
        near = []
        for d in DIRS:
            nx, ny = self.ahead(d)
            what = "neck" if d == OPP[self.dir] else self.cell(nx, ny)
            near.append(f"{d}={what}")
        recent = ", ".join(f"({x},{y})" for x, y in self.recent()[-6:]) or "none"
        return (
            f"Snake, {N}x{N} grid, columns 0-{N - 1} left to right, rows 0-{N - 1} top to bottom. "
            f"Head ({hx},{hy}) moving {self.dir}, length {len(self.snake)}. Food ({fx},{fy}), closer if moving: {toward}. "
            f"Neighbour cells: {', '.join(near)}. "
            f"Cells the head just came from, do not go back to them: {recent}. "
            f"Moving into wall or body loses."
        )

    RECENT = 10

    def recent(self) -> list[tuple[int, int]]:
        """Head cells since the last food, newest last."""
        return self.history[-self.RECENT :]

    def looping(self) -> bool:
        """The head is retracing its own path: it has come back to a cell it stood on
        within the last few moves, or it has gone three board-widths without eating,
        which is how a wide circle shows up."""
        h = self.recent()
        if len(h) >= 3 and h[-1] in h[:-1]:
            return True
        return len(self.history) > 3 * N or (len(h) >= 8 and len(set(h)) <= 4)

    def choose(self, probabilities: dict[str, float], safe: dict[str, float], mode: str = "legal") -> tuple[str, str]:
        """Returns (direction, reason). Modes: legal (filtered ranking with assistance), raw (top prediction, no reversing)."""
        order = sorted(probabilities, key=lambda d: -probabilities[d])
        legal = self.legal()
        if mode == "raw":
            choice = order[0]
            return (self.dir if choice == OPP[self.dir] else choice), (
                "reverse prediction: continuing straight" if choice == OPP[self.dir] else "top prediction"
            )
        if not legal:
            return order[0], "no legal move"
        if self.looping():
            # Prefer a legal move onto a cell it has not just stood on, nearest the food.
            recent = set(self.recent())
            return min(legal, key=lambda d: (self.ahead(d) in recent, self.distance(d))), "loop breaker: toward food"
        for d in order:
            if d in legal:
                # Keep straight when turning gets no closer to the food: otherwise the snake
                # zig-zags one cell at a time along the diagonal.
                if d != self.dir and self.dir in legal and self.distance(self.dir) <= self.distance(d):
                    return self.dir, "straight: turning gains nothing"
                return d, "model choice" if d == order[0] else "model choice among legal moves"
        return legal[0], "fallback"

    def step(self, d: str) -> bool:
        """Advance one tick. Returns False when the snake dies."""
        self.dir = d
        nx, ny = self.ahead(d)
        eats = (nx, ny) == self.food
        tail_moves = not eats
        body = self.snake[:-1] if tail_moves else self.snake
        if not (0 <= nx < N and 0 <= ny < N) or (nx, ny) in body:
            return False
        self.snake.insert(0, (nx, ny))
        if eats:
            self.score += 1
            self.history.clear()
            self.place_food()
        else:
            self.snake.pop()
            self.history.append((nx, ny))
        self.steps += 1
        return True
