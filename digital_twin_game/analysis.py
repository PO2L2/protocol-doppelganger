from __future__ import annotations

from dataclasses import dataclass

import pygame

from .actions import PlayerAction


@dataclass(frozen=True)
class ComboResult:
    name: str
    energy_bonus: float


class ComboTracker:
    """Recognises short action sequences without relying on a neural model."""

    def __init__(self) -> None:
        self.history: list[tuple[float, PlayerAction]] = []
        self.previous_action: PlayerAction | None = None
        self.combo_count = 0

    def record(self, timestamp: float, action: PlayerAction) -> ComboResult | None:
        combat_actions = {
            PlayerAction.RANGED_ATTACK,
            PlayerAction.MELEE_ATTACK,
            PlayerAction.DASH,
            PlayerAction.BLOCK,
            PlayerAction.HEAL,
        }
        if action not in combat_actions or action == self.previous_action:
            return None
        self.previous_action = action
        self.history.append((timestamp, action))
        self.history = [(time, item) for time, item in self.history if timestamp - time <= 2.6]
        actions = [item for _, item in self.history]

        result = None
        if self._ends_with(actions, (PlayerAction.RANGED_ATTACK, PlayerAction.DASH, PlayerAction.MELEE_ATTACK)):
            result = ComboResult("РАЗРЫВ ШАБЛОНА", 20)
        elif self._ends_with(actions, (PlayerAction.BLOCK, PlayerAction.DASH, PlayerAction.RANGED_ATTACK)):
            result = ComboResult("КОНТРПРОТОКОЛ", 18)
        elif self._ends_with(actions, (PlayerAction.MELEE_ATTACK, PlayerAction.DASH, PlayerAction.RANGED_ATTACK)):
            result = ComboResult("ОТВЕТНЫЙ ИМПУЛЬС", 16)
        elif self._ends_with(actions, (PlayerAction.RANGED_ATTACK, PlayerAction.BLOCK, PlayerAction.MELEE_ATTACK)):
            result = ComboResult("СМЕНА ВЕКТОРА", 16)

        if result:
            self.combo_count += 1
            self.history.clear()
            self.previous_action = None
        return result

    @staticmethod
    def _ends_with(actions: list[PlayerAction], pattern: tuple[PlayerAction, ...]) -> bool:
        return len(actions) >= len(pattern) and tuple(actions[-len(pattern) :]) == pattern


class PositionHeatmap:
    def __init__(self, columns: int = 24, rows: int = 12) -> None:
        self.columns = columns
        self.rows = rows
        self.cells = [[0 for _ in range(columns)] for _ in range(rows)]
        self.sample_count = 0

    def add(self, position: pygame.Vector2, bounds: pygame.Rect) -> None:
        normalized_x = (position.x - bounds.left) / max(1, bounds.width)
        normalized_y = (position.y - bounds.top) / max(1, bounds.height)
        column = max(0, min(self.columns - 1, int(normalized_x * self.columns)))
        row = max(0, min(self.rows - 1, int(normalized_y * self.rows)))
        self.cells[row][column] += 1
        self.sample_count += 1

    @property
    def maximum(self) -> int:
        return max((max(row) for row in self.cells), default=0)
