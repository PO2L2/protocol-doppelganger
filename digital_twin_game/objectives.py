from __future__ import annotations

from enum import Enum

import pygame

from .hidpi import draw as logical_draw

from .config import COLORS


class ObjectiveType(str, Enum):
    HOLD = "hold"
    COLLECT = "collect"
    PROTECT = "protect"


class TrainingObjective:
    def __init__(self, arena_id: int, bounds: pygame.Rect) -> None:
        self.arena_id = arena_id
        self.bounds = bounds
        self.kind = {1: ObjectiveType.HOLD, 2: ObjectiveType.COLLECT, 3: ObjectiveType.PROTECT}.get(arena_id, ObjectiveType.HOLD)
        self.progress = 0.0
        self.target = 10.0 if self.kind == ObjectiveType.HOLD else 5.0
        self.zone_position = pygame.Vector2(bounds.center)
        if self.kind == ObjectiveType.PROTECT:
            self.zone_position = pygame.Vector2(bounds.centerx, bounds.bottom - 82)
        self.zone_radius = 95
        self.core_health = 100.0
        self.cores = self._make_cores() if self.kind == ObjectiveType.COLLECT else []

    def _make_cores(self) -> list[tuple[pygame.Vector2, bool]]:
        left, top, width, height = self.bounds
        points = [
            (left + width * 0.18, top + height * 0.20),
            (left + width * 0.80, top + height * 0.18),
            (left + width * 0.50, top + height * 0.35),
            (left + width * 0.22, top + height * 0.78),
            (left + width * 0.78, top + height * 0.76),
        ]
        return [(pygame.Vector2(point), False) for point in points]

    def update(self, dt: float, player_position: pygame.Vector2, enemies: list) -> None:
        if self.kind == ObjectiveType.HOLD:
            if player_position.distance_to(self.zone_position) <= self.zone_radius:
                self.progress = min(self.target, self.progress + dt)
        elif self.kind == ObjectiveType.COLLECT:
            updated = []
            for position, collected in self.cores:
                if not collected and player_position.distance_to(position) < 34:
                    collected = True
                updated.append((position, collected))
            self.cores = updated
            self.progress = sum(1 for _, collected in self.cores if collected)
        elif self.kind == ObjectiveType.PROTECT:
            attackers = sum(1 for enemy in enemies if enemy.alive and enemy.position.distance_to(self.zone_position) < 88)
            self.core_health = max(0.0, self.core_health - attackers * 11 * dt)

    @property
    def complete(self) -> bool:
        if self.kind == ObjectiveType.PROTECT:
            return self.core_health > 0
        return self.progress >= self.target

    @property
    def failed(self) -> bool:
        return self.kind == ObjectiveType.PROTECT and self.core_health <= 0

    def status(self) -> str:
        if self.kind == ObjectiveType.HOLD:
            return f"Удержание точки: {self.progress:.1f}/{self.target:.0f} сек."
        if self.kind == ObjectiveType.COLLECT:
            return f"Энергетические ядра: {int(self.progress)}/{int(self.target)}"
        return f"Защита ядра: {self.core_health:.0f}%"

    def draw(self, surface: pygame.Surface) -> None:
        if self.kind == ObjectiveType.HOLD:
            color = COLORS["health"] if self.complete else COLORS["accent"]
            fill = tuple(max(8, channel // 7) for channel in color)
            outline = tuple(max(20, int(channel * 0.72)) for channel in color)
            logical_draw.circle(surface, fill, self.zone_position, self.zone_radius)
            logical_draw.circle(surface, outline, self.zone_position, self.zone_radius, 3)
        elif self.kind == ObjectiveType.COLLECT:
            for position, collected in self.cores:
                if collected:
                    continue
                logical_draw.circle(surface, COLORS["warning"], position, 11)
                logical_draw.circle(surface, COLORS["white"], position, 16, 2)
        else:
            ratio = self.core_health / 100
            logical_draw.circle(surface, (25, 57, 79), self.zone_position, 35)
            logical_draw.circle(surface, COLORS["accent"] if ratio > 0.3 else COLORS["enemy"], self.zone_position, 25)
            logical_draw.circle(surface, COLORS["white"], self.zone_position, 38, 2)
