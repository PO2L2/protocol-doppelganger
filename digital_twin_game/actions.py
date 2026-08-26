from enum import IntEnum


class PlayerAction(IntEnum):
    """The fixed 10-class output contract used by the neural network."""

    IDLE = 0
    APPROACH = 1
    RETREAT = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    RANGED_ATTACK = 5
    MELEE_ATTACK = 6
    DASH = 7
    BLOCK = 8
    HEAL = 9


ACTION_LABELS_RU = {
    PlayerAction.IDLE: "Ожидание",
    PlayerAction.APPROACH: "Сближение",
    PlayerAction.RETREAT: "Отступление",
    PlayerAction.STRAFE_LEFT: "Обход влево",
    PlayerAction.STRAFE_RIGHT: "Обход вправо",
    PlayerAction.RANGED_ATTACK: "Дальний удар",
    PlayerAction.MELEE_ATTACK: "Ближний удар",
    PlayerAction.DASH: "Рывок",
    PlayerAction.BLOCK: "Блок",
    PlayerAction.HEAL: "Лечение",
}
