from __future__ import annotations

from dataclasses import dataclass
import random

from .actions import PlayerAction
from .lab import SessionSummary


@dataclass
class TournamentFighter:
    name: str
    rates: dict[PlayerAction, float]
    health: float = 100.0
    energy: float = 100.0

    @classmethod
    def from_session(cls, session: SessionSummary, index: int) -> "TournamentFighter":
        return cls(f"Двойник {index}: {session.session_id}", {action: session.rate(action) for action in PlayerAction})

    @classmethod
    def demo(cls, name: str, aggressive: bool) -> "TournamentFighter":
        rates = {action: 0.04 for action in PlayerAction}
        if aggressive:
            rates[PlayerAction.APPROACH] = 0.25
            rates[PlayerAction.MELEE_ATTACK] = 0.24
            rates[PlayerAction.DASH] = 0.15
        else:
            rates[PlayerAction.RETREAT] = 0.18
            rates[PlayerAction.RANGED_ATTACK] = 0.27
            rates[PlayerAction.BLOCK] = 0.14
        total = sum(rates.values())
        return cls(name, {action: value / total for action, value in rates.items()})

    def choose(self) -> PlayerAction:
        actions = list(PlayerAction)
        weights = [max(0.001, self.rates.get(action, 0.0)) for action in actions]
        return random.choices(actions, weights=weights, k=1)[0]


class TournamentMatch:
    def __init__(self, first: TournamentFighter, second: TournamentFighter) -> None:
        self.first = first
        self.second = second
        self.timer = 0.4
        self.round_time = 0.0
        self.last_first = PlayerAction.IDLE
        self.last_second = PlayerAction.IDLE
        self.log: list[str] = []

    @property
    def finished(self) -> bool:
        return self.first.health <= 0 or self.second.health <= 0 or self.round_time >= 45

    @property
    def winner(self) -> TournamentFighter:
        return self.first if self.first.health >= self.second.health else self.second

    def update(self, dt: float) -> None:
        if self.finished:
            return
        self.round_time += dt
        self.timer -= dt
        if self.timer > 0:
            return
        self.timer = 0.48
        self.last_first = self.first.choose()
        self.last_second = self.second.choose()
        first_damage = self._damage(self.last_first, self.last_second)
        second_damage = self._damage(self.last_second, self.last_first)
        self.second.health = max(0.0, self.second.health - first_damage)
        self.first.health = max(0.0, self.first.health - second_damage)
        if first_damage or second_damage:
            self.log.append(f"{self.last_first.name} / {self.last_second.name}")
            self.log = self.log[-4:]

    @staticmethod
    def _damage(action: PlayerAction, opponent: PlayerAction) -> float:
        if action not in (PlayerAction.RANGED_ATTACK, PlayerAction.MELEE_ATTACK):
            return 0.0
        base = 9.0 if action == PlayerAction.RANGED_ATTACK else 12.0
        if opponent == PlayerAction.BLOCK:
            base *= 0.25
        elif opponent == PlayerAction.DASH:
            base *= 0.35
        elif action == PlayerAction.MELEE_ATTACK and opponent == PlayerAction.RETREAT:
            base *= 0.45
        elif action == PlayerAction.RANGED_ATTACK and opponent == PlayerAction.APPROACH:
            base *= 1.25
        return base

