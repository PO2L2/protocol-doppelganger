from __future__ import annotations

import pygame

from .actions import PlayerAction
from .arena import Arena
from .config import HEIGHT, WIDTH
from .entities import Enemy, Player
from .model_interface import FEATURE_NAMES, validate_features


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def classify_movement_action(
    player: Player,
    opponent: Enemy | None,
    explicit_action: PlayerAction | None,
) -> PlayerAction:
    if explicit_action is not None:
        return explicit_action
    if player.block_active:
        return PlayerAction.BLOCK
    if player.velocity.length_squared() < 100:
        return PlayerAction.IDLE
    if opponent is None:
        return PlayerAction.APPROACH
    to_opponent = opponent.position - player.position
    if to_opponent.length_squared() < 1:
        return PlayerAction.IDLE
    facing_target = to_opponent.normalize()
    movement = player.velocity.normalize()
    forward = movement.dot(facing_target)
    cross = facing_target.x * movement.y - facing_target.y * movement.x
    if forward > 0.48:
        return PlayerAction.APPROACH
    if forward < -0.48:
        return PlayerAction.RETREAT
    return PlayerAction.STRAFE_RIGHT if cross > 0 else PlayerAction.STRAFE_LEFT


def build_feature_vector(player: Player, opponent: Enemy | None, arena: Arena) -> tuple[float, ...]:
    if opponent is None:
        opponent_position = player.position
        opponent_velocity = pygame.Vector2()
        opponent_health = 0.0
        opponent_attacking = 0.0
    else:
        opponent_position = opponent.position
        opponent_velocity = opponent.velocity
        opponent_health = opponent.health / opponent.max_health
        opponent_attacking = 1.0 if opponent.attack_windup > 0 else 0.0

    relative = opponent_position - player.position
    max_distance = pygame.Vector2(arena.bounds.size).length()
    active_packs = [pack for pack in arena.health_packs if pack.active]
    if active_packs:
        pack = min(active_packs, key=lambda item: item.position.distance_squared_to(player.position))
        pack_relative = pack.position - player.position
        pack_distance = clamp(pack_relative.length() / max_distance, 0, 1)
        pack_x = clamp(pack_relative.x / WIDTH)
        pack_y = clamp(pack_relative.y / HEIGHT)
    else:
        pack_distance, pack_x, pack_y = 1.0, 0.0, 0.0

    values = (
        clamp(player.health / player.max_health, 0, 1),
        clamp(player.energy / player.max_energy, 0, 1),
        clamp(opponent_health, 0, 1),
        clamp(relative.length() / max_distance, 0, 1),
        clamp(relative.x / WIDTH),
        clamp(relative.y / HEIGHT),
        clamp(player.velocity.x / 300),
        clamp(player.velocity.y / 300),
        clamp(opponent_velocity.x / 300),
        clamp(opponent_velocity.y / 300),
        clamp(player.ranged_cooldown / 0.34, 0, 1),
        clamp(player.melee_cooldown / 0.62, 0, 1),
        clamp(player.dash_cooldown / 1.15, 0, 1),
        1.0 if player.block_active else 0.0,
        1.0 if player.heal_charges > 0 else 0.0,
        clamp((player.position.x - arena.bounds.left) / arena.bounds.width, 0, 1),
        clamp((arena.bounds.right - player.position.x) / arena.bounds.width, 0, 1),
        clamp((player.position.y - arena.bounds.top) / arena.bounds.height, 0, 1),
        clamp((arena.bounds.bottom - player.position.y) / arena.bounds.height, 0, 1),
        pack_distance,
        pack_x,
        pack_y,
        opponent_attacking,
        clamp(player.recent_damage_timer / 0.8, 0, 1),
        clamp((int(player.last_action) / 9.0) * 2.0 - 1.0),
    )
    assert len(values) == len(FEATURE_NAMES)
    return validate_features(values)
