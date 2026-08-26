from __future__ import annotations

from enum import Enum


class UpgradeType(str, Enum):
    HEALTH = "health"
    ENERGY = "energy"
    SPEED = "speed"
    DAMAGE = "damage"
    COOLDOWN = "cooldown"
    HEAL = "heal"


UPGRADE_INFO: dict[UpgradeType, tuple[str, str]] = {
    UpgradeType.HEALTH: ("Усиленный корпус", "+20 к максимальному здоровью"),
    UpgradeType.ENERGY: ("Расширенный накопитель", "+20 к максимальной энергии"),
    UpgradeType.SPEED: ("Сервоприводы", "+12% к скорости движения"),
    UpgradeType.DAMAGE: ("Перегрузка оружия", "+15% к любому наносимому урону"),
    UpgradeType.COOLDOWN: ("Контур охлаждения", "Атаки восстанавливаются на 14% быстрее"),
    UpgradeType.HEAL: ("Резервный наноблок", "+1 заряд лечения на каждой арене"),
}

