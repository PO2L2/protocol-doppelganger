"""Arena-specific mini challenges that deliberately collect rare actions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChallengeGoal:
    event: str
    label: str
    target: int


GOALS = {
    1: (ChallengeGoal("ranged_hit", "Попадания издалека", 5), ChallengeGoal("block", "Успешные блоки", 3)),
    2: (ChallengeGoal("dash", "Рывки", 4), ChallengeGoal("perfect_dodge", "Идеальные уклонения", 1)),
    3: (ChallengeGoal("melee_hit", "Ближние попадания", 4), ChallengeGoal("heal", "Эффективное лечение", 1)),
}


class CalibrationChallenge:
    def __init__(self, arena_id: int) -> None:
        self.arena_id = arena_id
        self.goals = GOALS.get(arena_id, ())
        self.counts = {goal.event: 0 for goal in self.goals}

    def record(self, event: str, amount: int = 1) -> None:
        if event in self.counts:
            self.counts[event] = min(self.target(event), self.counts[event] + amount)

    def target(self, event: str) -> int:
        return next((goal.target for goal in self.goals if goal.event == event), 0)

    @property
    def complete(self) -> bool:
        return bool(self.goals) and all(self.counts[goal.event] >= goal.target for goal in self.goals)

    def lines(self) -> list[tuple[str, int, int]]:
        return [(goal.label, self.counts[goal.event], goal.target) for goal in self.goals]
