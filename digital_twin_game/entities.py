from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import pygame

from .hidpi import draw as logical_draw

from .actions import PlayerAction
from .arena import Arena
from .behavior import BehaviorProfile
from .config import COLORS, PLAYER_MAX_ENERGY, PLAYER_MAX_HEALTH, PLAYER_RADIUS, PLAYER_SPEED
from .weapons import WeaponType, weapon_spec


def safe_normalize(vector: pygame.Vector2) -> pygame.Vector2:
    return vector.normalize() if vector.length_squared() > 0.0001 else pygame.Vector2()


def segment_intersects_circle(
    start: pygame.Vector2,
    end: pygame.Vector2,
    center: pygame.Vector2,
    radius: float,
) -> bool:
    segment = end - start
    if not segment.length_squared():
        return start.distance_squared_to(center) <= radius * radius
    ratio = max(0.0, min(1.0, (center - start).dot(segment) / segment.length_squared()))
    closest = start + segment * ratio
    return closest.distance_squared_to(center) <= radius * radius


@dataclass
class Projectile:
    position: pygame.Vector2
    velocity: pygame.Vector2
    damage: float
    owner: str
    color: tuple[int, int, int]
    radius: int = 6
    lifetime: float = 2.0
    impact_position: pygame.Vector2 | None = None
    previous_position: pygame.Vector2 = field(init=False)

    def __post_init__(self) -> None:
        self.previous_position = self.position.copy()

    def update(self, dt: float, arena: Arena) -> bool:
        previous = self.position.copy()
        self.previous_position = previous
        self.position += self.velocity * dt
        self.lifetime -= dt
        if not arena.bounds.collidepoint(self.position):
            return False
        destructible_hit = arena.hit_destructible(previous, self.position, self.damage)
        if destructible_hit is not None:
            self.impact_position = destructible_hit
            return False
        if any(obstacle.clipline(previous, self.position) for obstacle in arena.obstacles):
            return False
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface) -> None:
        logical_draw.circle(surface, self.color, self.position, self.radius + 4, 1)
        logical_draw.circle(surface, self.color, self.position, self.radius)


@dataclass
class SlashEffect:
    position: pygame.Vector2
    color: tuple[int, int, int]
    radius: float
    lifetime: float = 0.18

    def update(self, dt: float) -> bool:
        self.lifetime -= dt
        self.radius += 120 * dt
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface) -> None:
        ratio = max(0.0, min(1.0, self.lifetime / 0.18))
        color = tuple(int(channel * (0.3 + ratio * 0.7)) for channel in self.color)
        logical_draw.circle(surface, color, self.position, int(self.radius), 4)


class Player:
    def __init__(self, position: tuple[float, float]) -> None:
        self.position = pygame.Vector2(position)
        self.velocity = pygame.Vector2()
        self.radius = PLAYER_RADIUS
        self.max_health = PLAYER_MAX_HEALTH
        self.health = self.max_health
        self.max_energy = PLAYER_MAX_ENERGY
        self.energy = self.max_energy
        self.ranged_cooldown = 0.0
        self.melee_cooldown = 0.0
        self.dash_cooldown = 0.0
        self.block_active = False
        self.heal_charges = 1
        self.last_action = PlayerAction.IDLE
        self.recent_damage_timer = 0.0
        self.flash_timer = 0.0
        self.facing = pygame.Vector2(1, 0)
        self.reflect_timer = 0.0
        self.shield_timer = 0.0
        self.ability_lock_timer = 0.0
        self.last_used_ability: str | None = None
        self.weapon = WeaponType.PULSE
        self.speed_multiplier = 1.0
        self.damage_multiplier = 1.0
        self.cooldown_multiplier = 1.0

    def set_weapon(self, weapon: WeaponType | str) -> None:
        self.weapon = WeaponType(weapon)

    @property
    def alive(self) -> bool:
        return self.health > 0

    def update(self, dt: float, movement: pygame.Vector2, arena: Arena) -> None:
        self.ranged_cooldown = max(0.0, self.ranged_cooldown - dt)
        self.melee_cooldown = max(0.0, self.melee_cooldown - dt)
        self.dash_cooldown = max(0.0, self.dash_cooldown - dt)
        self.recent_damage_timer = max(0.0, self.recent_damage_timer - dt)
        self.flash_timer = max(0.0, self.flash_timer - dt)
        self.reflect_timer = max(0.0, self.reflect_timer - dt)
        self.shield_timer = max(0.0, self.shield_timer - dt)
        self.ability_lock_timer = max(0.0, self.ability_lock_timer - dt)
        self.energy = min(self.max_energy, self.energy + (18 if not self.block_active else 2) * dt)
        direction = safe_normalize(movement)
        self.velocity = direction * PLAYER_SPEED * self.speed_multiplier
        self.position = arena.move_circle(self.position, self.velocity * dt, self.radius)

    def dash(self, direction: pygame.Vector2, arena: Arena) -> bool:
        if self.dash_cooldown > 0 or self.energy < 20:
            return False
        direction = safe_normalize(direction if direction.length_squared() else self.facing)
        self.position = arena.move_circle(self.position, direction * 112, self.radius)
        self.energy -= 20
        self.dash_cooldown = 1.15
        self.last_action = PlayerAction.DASH
        return True

    def fire(self, target: pygame.Vector2) -> list[Projectile]:
        spec = weapon_spec(self.weapon)
        if self.ranged_cooldown > 0 or self.energy < spec.energy_cost:
            return []
        direction = safe_normalize(target - self.position)
        if not direction.length_squared():
            return []
        self.facing = direction
        self.energy -= spec.energy_cost
        self.ranged_cooldown = spec.ranged_cooldown * self.cooldown_multiplier
        self.last_action = PlayerAction.RANGED_ATTACK
        if spec.projectile_count == 1:
            angles = [0.0]
        else:
            step = spec.spread_degrees / max(1, spec.projectile_count - 1)
            angles = [-spec.spread_degrees / 2 + step * index for index in range(spec.projectile_count)]
        return [
            Projectile(
                self.position + direction.rotate(angle) * 25,
                direction.rotate(angle) * spec.projectile_speed,
                spec.ranged_damage * self.damage_multiplier,
                "player",
                COLORS["player"],
                radius=5 if self.weapon == WeaponType.RAIL else 6,
            )
            for angle in angles
        ]

    def melee(self, target: pygame.Vector2, enemies: list[Enemy], line_of_sight=None) -> tuple[bool, SlashEffect | None]:
        spec = weapon_spec(self.weapon)
        melee_cost = spec.melee_energy_cost
        if self.melee_cooldown > 0 or self.energy < melee_cost:
            return False, None
        direction = safe_normalize(target - self.position)
        candidates = [
            enemy
            for enemy in enemies
            if enemy.alive
            and enemy.position.distance_to(self.position) <= spec.melee_range + enemy.radius
            and (line_of_sight is None or line_of_sight(self.position, enemy.position))
        ]
        attack_facing = direction if direction.length_squared() else self.facing
        aimed_candidates = [
            enemy
            for enemy in candidates
            if not (enemy.position - self.position).length_squared()
            or safe_normalize(enemy.position - self.position).dot(attack_facing) > 0.15
        ]
        if candidates and not aimed_candidates:
            nearest = min(candidates, key=lambda enemy: enemy.position.distance_squared_to(self.position))
            direction = safe_normalize(nearest.position - self.position)
        if direction.length_squared():
            self.facing = direction
        self.energy -= melee_cost
        self.melee_cooldown = spec.melee_cooldown * self.cooldown_multiplier
        self.last_action = PlayerAction.MELEE_ATTACK
        hit = False
        for enemy in candidates:
            offset = enemy.position - self.position
            touching = offset.length() <= self.radius + enemy.radius + 10
            in_attack_arc = not offset.length_squared() or safe_normalize(offset).dot(self.facing) > 0.05
            if touching or in_attack_arc:
                enemy.take_damage(spec.melee_damage * self.damage_multiplier, self.position, melee=True)
                hit = True
        return hit, SlashEffect(self.position.copy(), COLORS["player"], 42)

    def set_block(self, enabled: bool, dt: float) -> None:
        self.block_active = enabled and self.energy > 0
        if self.block_active:
            self.energy = max(0.0, self.energy - 25 * dt)
            self.last_action = PlayerAction.BLOCK

    def heal(self) -> bool:
        if self.heal_charges <= 0 or self.health >= self.max_health or self.energy < 10:
            return False
        self.heal_charges -= 1
        self.health = min(self.max_health, self.health + 36)
        self.energy -= 10
        self.last_action = PlayerAction.HEAL
        return True

    def take_damage(self, amount: float) -> None:
        if self.dash_cooldown > 0.92:
            return
        multiplier = 0.28 if self.block_active else 1.0
        if self.shield_timer > 0:
            multiplier *= 0.35
        applied = amount * multiplier
        self.health = max(0.0, self.health - applied)
        self.recent_damage_timer = 0.8
        self.flash_timer = 0.12

    def draw(self, surface: pygame.Surface) -> None:
        color = COLORS["white"] if self.flash_timer > 0 else COLORS["player"]
        logical_draw.circle(surface, COLORS["player_glow"], self.position, self.radius + 7)
        logical_draw.circle(surface, color, self.position, self.radius)
        nose = self.position + self.facing * (self.radius + 8)
        logical_draw.line(surface, COLORS["white"], self.position, nose, 4)
        if self.block_active:
            logical_draw.circle(surface, COLORS["accent"], self.position, self.radius + 13, 3)
        if self.shield_timer > 0:
            logical_draw.circle(surface, COLORS["warning"], self.position, self.radius + 19, 3)
        if self.reflect_timer > 0:
            logical_draw.circle(surface, COLORS["white"], self.position, self.radius + 25, 2)


@dataclass
class TrapHazard:
    position: pygame.Vector2
    owner: str
    lifetime: float = 9.0
    radius: float = 34.0
    damage: float = 24.0

    def update(self, dt: float, player: Player, enemies: list[Enemy]) -> tuple[bool, pygame.Vector2 | None, float]:
        self.lifetime -= dt
        if self.owner == "player":
            for enemy in enemies:
                if enemy.alive and enemy.position.distance_to(self.position) < self.radius:
                    enemy.take_damage(self.damage, self.position)
                    return False, enemy.position.copy(), self.damage
        elif player.alive and player.position.distance_to(self.position) < self.radius:
            player.take_damage(self.damage)
            return False, player.position.copy(), self.damage
        return self.lifetime > 0, None, 0.0

    def draw(self, surface: pygame.Surface) -> None:
        color = COLORS["player"] if self.owner == "player" else COLORS["enemy"]
        pulse = 7 + int(math.sin(self.lifetime * 8) * 2)
        logical_draw.circle(surface, color, self.position, pulse, 2)
        logical_draw.circle(surface, tuple(component // 2 for component in color), self.position, int(self.radius), 1)


@dataclass
class TurretHazard:
    position: pygame.Vector2
    lifetime: float = 12.0
    cooldown: float = 0.8

    def update(self, dt: float, player: Player, projectiles: list[Projectile]) -> bool:
        self.lifetime -= dt
        self.cooldown -= dt
        if self.cooldown <= 0 and player.alive:
            direction = safe_normalize(player.position - self.position)
            if direction.length_squared():
                projectiles.append(Projectile(self.position + direction * 18, direction * 410, 8, "enemy", COLORS["warning"], radius=5))
                self.cooldown = 1.15
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface) -> None:
        logical_draw.circle(surface, (89, 58, 33), self.position, 16)
        logical_draw.circle(surface, COLORS["warning"], self.position, 11)
        logical_draw.circle(surface, COLORS["enemy"], self.position, 20, 2)


@dataclass
class Decoy:
    position: pygame.Vector2
    lifetime: float = 4.5

    def update(self, dt: float) -> bool:
        self.lifetime -= dt
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface) -> None:
        ratio = max(0.0, min(1.0, self.lifetime / 4.5))
        strength = 0.25 + ratio * 0.35
        body = tuple(int(channel * strength) for channel in COLORS["player"])
        outline = tuple(int(channel * strength) for channel in COLORS["white"])
        logical_draw.circle(surface, body, self.position, PLAYER_RADIUS)
        logical_draw.circle(surface, outline, self.position, PLAYER_RADIUS + 8, 2)


class Enemy:
    def __init__(self, position: tuple[float, float], kind: str = "hunter", twin_profile: BehaviorProfile | None = None) -> None:
        self.position = pygame.Vector2(position)
        self.velocity = pygame.Vector2()
        self.radius = 18 if kind != "twin" else 20
        self.kind = kind
        health_by_kind = {
            "hunter": 72.0,
            "shooter": 66.0,
            "assault": 82.0,
            "sniper": 58.0,
            "shield": 115.0,
            "teleporter": 70.0,
            "engineer": 78.0,
            "copier": 88.0,
            "twin": 165.0,
        }
        self.max_health = health_by_kind.get(kind, 72.0)
        self.health = self.max_health
        self.attack_cooldown = random.uniform(0.1, 0.6)
        self.attack_windup = 0.0
        self.attack_windup_total = 0.0
        self.queued_attack: str | None = None
        self.queued_direction = pygame.Vector2(-1, 0)
        self.decision_timer = 0.0
        self.attack_flash = 0.0
        self.recent_damage = 0.0
        self.strafe_sign = random.choice((-1, 1))
        self.profile = twin_profile
        self.facing = pygame.Vector2(-1, 0)
        self.teleport_cooldown = random.uniform(2.5, 4.0)
        self.deploy_cooldown = random.uniform(2.0, 3.5)
        self.slow_timer = 0.0
        self.phase = 1
        self.copied_ability: str | None = None
        self.copy_use_cooldown = 0.0

    @property
    def alive(self) -> bool:
        return self.health > 0

    @property
    def color(self) -> tuple[int, int, int]:
        colors = {
            "assault": COLORS["enemy"],
            "sniper": COLORS["warning"],
            "shield": COLORS["accent"],
            "teleporter": (63, 211, 224),
            "engineer": (232, 172, 68),
            "copier": (232, 128, 226),
            "twin": COLORS["twin"],
        }
        return colors.get(self.kind, COLORS["enemy"])

    def take_damage(self, amount: float, source_position: pygame.Vector2 | None = None, *, melee: bool = False) -> float:
        applied = amount
        if self.kind == "shield" and source_position is not None:
            toward_source = safe_normalize(source_position - self.position)
            if toward_source.dot(self.facing) > 0.2:
                applied *= 0.48 if melee else 0.22
        self.health = max(0.0, self.health - applied)
        self.recent_damage = 0.12
        return applied

    def update(
        self,
        dt: float,
        player: Player,
        arena: Arena,
        projectiles: list[Projectile],
        predicted_player_action: PlayerAction | None = None,
        synchronization: float = 0.0,
        hazards: list | None = None,
        forced_target: pygame.Vector2 | None = None,
        twin_phase: int = 1,
    ) -> None:
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.attack_flash = max(0.0, self.attack_flash - dt)
        self.recent_damage = max(0.0, self.recent_damage - dt)
        self.decision_timer -= dt
        self.teleport_cooldown -= dt
        self.deploy_cooldown -= dt
        self.slow_timer = max(0.0, self.slow_timer - dt)
        self.copy_use_cooldown = max(0.0, self.copy_use_cooldown - dt)
        self.phase = twin_phase
        target_position = forced_target if forced_target is not None else player.position
        offset = target_position - self.position
        distance = max(1.0, offset.length())
        direction = offset / distance
        self.facing = direction

        if self.attack_windup > 0:
            self.attack_windup -= dt
            if self.attack_windup <= 0:
                self._release_attack(player, projectiles, synchronization, arena=arena)
            if self.queued_attack == "melee":
                self.velocity.update(0, 0)
                return

        preferred_by_kind = {
            "hunter": 155,
            "shooter": 330,
            "assault": 78,
            "sniper": 500,
            "shield": 125,
            "teleporter": 260,
            "engineer": 340,
            "copier": 135,
        }
        speed_by_kind = {
            "hunter": 175,
            "shooter": 165,
            "assault": 250,
            "sniper": 125,
            "shield": 138,
            "teleporter": 185,
            "engineer": 150,
            "copier": 205,
        }
        preferred = preferred_by_kind.get(self.kind, 330)
        speed = speed_by_kind.get(self.kind, 175) if self.kind != "twin" else 220 + synchronization * 28
        ranged_chance = 0.65 if self.kind in ("shooter", "engineer") else 0.22
        if self.kind == "sniper":
            ranged_chance = 1.0
        if self.slow_timer > 0:
            speed *= 0.48
        if self.profile and self.kind == "twin":
            preferred = 135 + self.profile.preferred_range * 350
            ranged = self.profile.action_rates[PlayerAction.RANGED_ATTACK]
            melee = self.profile.action_rates[PlayerAction.MELEE_ATTACK]
            ranged_chance = 0.3 + 0.55 * (ranged / max(0.01, ranged + melee))
            speed += self.profile.mobility * 25

        perpendicular = pygame.Vector2(-direction.y, direction.x) * self.strafe_sign
        if distance > preferred + 45:
            move = direction
        elif distance < preferred - 45:
            move = -direction
        else:
            move = perpendicular

        # The twin consumes the predicted action returned by the shared model
        # interface and adjusts its steering according to the active phase.
        if self.kind == "twin" and predicted_player_action is not None:
            if twin_phase == 1:  # mirror the player's preferred behaviour
                if predicted_player_action in (PlayerAction.APPROACH, PlayerAction.MELEE_ATTACK):
                    move = direction
                elif predicted_player_action in (PlayerAction.RETREAT, PlayerAction.HEAL):
                    move = -direction
                elif predicted_player_action == PlayerAction.STRAFE_LEFT:
                    move = pygame.Vector2(-direction.y, direction.x)
                elif predicted_player_action == PlayerAction.STRAFE_RIGHT:
                    move = pygame.Vector2(direction.y, -direction.x)
            elif twin_phase == 2:  # counter the prediction
                if predicted_player_action in (PlayerAction.RETREAT, PlayerAction.HEAL):
                    move = direction
                elif predicted_player_action in (PlayerAction.APPROACH, PlayerAction.MELEE_ATTACK):
                    move = -direction if distance < 115 else perpendicular
                elif predicted_player_action == PlayerAction.STRAFE_LEFT:
                    move = pygame.Vector2(direction.y, -direction.x)
                elif predicted_player_action == PlayerAction.STRAFE_RIGHT:
                    move = pygame.Vector2(-direction.y, direction.x)
                elif predicted_player_action == PlayerAction.RANGED_ATTACK:
                    move = direction + perpendicular * 0.45
            else:  # distorted mixture of imitation and counters
                move = safe_normalize(move + perpendicular * random.uniform(-1.1, 1.1))
                speed *= 1.18

        if self.kind == "teleporter" and self.teleport_cooldown <= 0:
            self._teleport(arena)
            self.teleport_cooldown = random.uniform(3.2, 4.8)
        if self.kind == "engineer" and hazards is not None and self.deploy_cooldown <= 0:
            if sum(isinstance(item, TurretHazard) for item in hazards) < 2 and random.random() < 0.5:
                hazards.append(TurretHazard(self.position.copy()))
            else:
                hazards.append(TrapHazard(self.position.copy(), "enemy"))
            self.deploy_cooldown = random.uniform(4.0, 6.0)

        if self.decision_timer <= 0:
            self.decision_timer = random.uniform(0.45, 1.0)
            if random.random() < 0.35:
                self.strafe_sign *= -1
        self.velocity = safe_normalize(move) * speed
        self.position = arena.move_circle(self.position, self.velocity * dt, self.radius)

        if self.attack_windup > 0 or self.attack_cooldown > 0 or not arena.has_line_of_sight(self.position, player.position):
            return
        if distance < 68:
            duration = 0.25 if self.kind == "assault" else (0.38 if self.kind == "twin" else 0.48)
            self._queue_attack("melee", direction, duration)
        elif distance < 560 and (
            random.random() < ranged_chance
            or (self.kind == "twin" and predicted_player_action in (PlayerAction.HEAL, PlayerAction.RANGED_ATTACK))
        ):
            spread = 0.02 if self.kind == "twin" else 0.055
            angle = math.atan2(direction.y, direction.x) + random.uniform(-spread, spread)
            duration = 1.15 if self.kind == "sniper" else (0.44 if self.kind == "twin" else 0.58)
            self._queue_attack("ranged", pygame.Vector2(math.cos(angle), math.sin(angle)), duration)

    def _teleport(self, arena: Arena) -> None:
        candidates = [
            pygame.Vector2(arena.bounds.left + 90, arena.bounds.top + 90),
            pygame.Vector2(arena.bounds.right - 90, arena.bounds.top + 90),
            pygame.Vector2(arena.bounds.left + 90, arena.bounds.bottom - 90),
            pygame.Vector2(arena.bounds.right - 90, arena.bounds.bottom - 90),
            pygame.Vector2(arena.bounds.center),
        ]
        random.shuffle(candidates)
        for candidate in candidates:
            circle = pygame.Rect(candidate.x - self.radius, candidate.y - self.radius, self.radius * 2, self.radius * 2)
            if not any(circle.colliderect(obstacle) for obstacle in arena.collision_rects):
                self.position = arena.place_circle(candidate, self.radius)
                return

    def _queue_attack(self, kind: str, direction: pygame.Vector2, duration: float) -> None:
        self.queued_attack = kind
        self.queued_direction = direction.copy()
        self.attack_windup = duration
        self.attack_windup_total = duration

    def _release_attack(
        self,
        player: Player,
        projectiles: list[Projectile],
        synchronization: float,
        arena: Arena | None = None,
    ) -> None:
        if self.queued_attack == "melee":
            visible = arena is None or arena.has_line_of_sight(self.position, player.position)
            if self.position.distance_to(player.position) < 82 and visible:
                base_damage = 19 if self.kind == "twin" else (13 if self.kind == "assault" else 16)
                player.take_damage(base_damage * (1.0 + synchronization * 0.12))
                if self.kind == "copier":
                    player.ability_lock_timer = max(player.ability_lock_timer, 3.5)
                    self.copied_ability = player.last_used_ability
            self.attack_cooldown = 0.42 if self.kind == "assault" else 0.82
        elif self.queued_attack == "ranged":
            base_damage = 27 if self.kind == "sniper" else (12 if self.kind == "twin" else 10)
            damage = base_damage * (1.0 + synchronization * 0.12)
            speed = 780 if self.kind == "sniper" else 475
            projectiles.append(
                Projectile(self.position + self.queued_direction * 24, self.queued_direction * speed, damage, "enemy", self.color)
            )
            self.attack_cooldown = 1.8 if self.kind == "sniper" else (0.72 if self.kind == "twin" else 1.02)
        self.attack_flash = 0.14
        self.queued_attack = None

    def draw(self, surface: pygame.Surface, bounds: pygame.Rect | None = None) -> None:
        if self.attack_windup > 0 and self.attack_windup_total > 0:
            progress = 1.0 - self.attack_windup / self.attack_windup_total
            warning_color = COLORS["warning"] if progress < 0.7 else COLORS["enemy"]
            if self.queued_attack == "melee":
                logical_draw.circle(surface, warning_color, self.position, 72, 2 + int(progress * 3))
        color = COLORS["white"] if self.recent_damage > 0 else self.color
        logical_draw.circle(surface, tuple(max(0, component // 3) for component in self.color), self.position, self.radius + 7)
        logical_draw.circle(surface, color, self.position, self.radius)
        nose = self.position + self.facing * (self.radius + 7)
        logical_draw.line(surface, COLORS["white"], self.position, nose, 3)
        if self.kind == "twin":
            phase_colors = {1: COLORS["player"], 2: COLORS["twin"], 3: COLORS["enemy"]}
            logical_draw.circle(surface, phase_colors.get(self.phase, COLORS["twin"]), self.position, self.radius + 13, 2 + self.phase // 2)
        elif self.kind == "shield":
            normal = safe_normalize(self.facing)
            perpendicular = pygame.Vector2(-normal.y, normal.x)
            center = self.position + normal * 24
            logical_draw.line(surface, COLORS["accent"], center - perpendicular * 24, center + perpendicular * 24, 7)
        elif self.kind == "teleporter":
            logical_draw.circle(surface, COLORS["accent"], self.position, self.radius + 12, 2)
        elif self.kind == "engineer":
            logical_draw.rect(surface, COLORS["warning"], (self.position.x - 7, self.position.y - 7, 14, 14), 2)
        elif self.kind == "copier":
            logical_draw.circle(surface, COLORS["white"], self.position, 7, 2)
        width = 44
        ratio = self.health / self.max_health
        logical_draw.rect(surface, (45, 22, 33), (self.position.x - width / 2, self.position.y - 33, width, 5), border_radius=2)
        logical_draw.rect(surface, self.color, (self.position.x - width / 2, self.position.y - 33, width * ratio, 5), border_radius=2)
