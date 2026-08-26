from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WeaponType(str, Enum):
    PULSE = "pulse"
    SHOTGUN = "shotgun"
    RAIL = "rail"
    BLADES = "blades"


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    description: str
    ranged_damage: float
    projectile_speed: float
    projectile_count: int
    spread_degrees: float
    ranged_cooldown: float
    energy_cost: float
    melee_damage: float
    melee_range: float


WEAPON_SPECS: dict[WeaponType, WeaponSpec] = {
    WeaponType.PULSE: WeaponSpec(
        "Импульсный автомат", "Стабильная универсальная стрельба", 12, 660, 1, 0, 0.34, 8, 23, 78
    ),
    WeaponType.SHOTGUN: WeaponSpec(
        "Энергетический дробовик", "Пять зарядов с большим разбросом", 7, 590, 5, 24, 0.82, 16, 25, 82
    ),
    WeaponType.RAIL: WeaponSpec(
        "Рельсовая винтовка", "Медленный, но очень мощный выстрел", 31, 900, 1, 0, 0.92, 19, 20, 72
    ),
    WeaponType.BLADES: WeaponSpec(
        "Парные клинки", "Быстрый ближний бой и слабые заряды", 7, 700, 1, 0, 0.26, 6, 36, 98
    ),
}


def weapon_spec(weapon: WeaponType | str) -> WeaponSpec:
    return WEAPON_SPECS[WeaponType(weapon)]

