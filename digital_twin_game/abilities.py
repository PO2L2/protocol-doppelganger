from __future__ import annotations

from enum import Enum


class AbilityType(str, Enum):
    TRAP = "trap"
    WAVE = "wave"
    REFLECT = "reflect"
    TELEPORT = "teleport"
    SHIELD = "shield"
    DECOY = "decoy"
    SLOW = "slow"


ABILITY_INFO: dict[AbilityType, tuple[str, str, float]] = {
    AbilityType.TRAP: ("Ловушка", "Мина, срабатывающая под противником", 6.0),
    AbilityType.WAVE: ("Энергетическая волна", "Урон всем ближайшим врагам", 7.5),
    AbilityType.REFLECT: ("Отражение", "Возвращает вражеские снаряды", 9.0),
    AbilityType.TELEPORT: ("Телепортация", "Перемещение в направлении прицела", 8.0),
    AbilityType.SHIELD: ("Временный щит", "Сильно уменьшает получаемый урон", 10.0),
    AbilityType.DECOY: ("Ложный клон", "Отвлекает противников", 9.0),
    AbilityType.SLOW: ("Замедление", "Замедляет всех противников", 11.0),
}


class AbilitySystem:
    def __init__(self, selected: list[AbilityType] | None = None) -> None:
        self.selected = list(selected or [AbilityType.TRAP, AbilityType.WAVE, AbilityType.SHIELD])[:3]
        self.cooldowns = {ability: 0.0 for ability in AbilityType}

    def update(self, dt: float) -> None:
        for ability in self.cooldowns:
            self.cooldowns[ability] = max(0.0, self.cooldowns[ability] - dt)

    def activate(self, slot: int, locked: bool = False) -> AbilityType | None:
        if locked or slot < 0 or slot >= len(self.selected):
            return None
        ability = self.selected[slot]
        if self.cooldowns[ability] > 0:
            return None
        self.cooldowns[ability] = ABILITY_INFO[ability][2]
        return ability

    def cooldown_ratio(self, ability: AbilityType) -> float:
        maximum = ABILITY_INFO[ability][2]
        return self.cooldowns[ability] / maximum if maximum else 0.0

