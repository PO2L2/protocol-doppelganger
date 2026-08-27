from __future__ import annotations

from dataclasses import dataclass

import pygame

from .hidpi import draw as logical_draw

from .config import COLORS, HEIGHT, PLAY_RECT, WIDTH


@dataclass
class HealthPack:
    position: pygame.Vector2
    active: bool = True


@dataclass
class DestructibleObject:
    rect: pygame.Rect
    health: float = 42.0
    maximum_health: float = 42.0

    @property
    def position(self) -> pygame.Vector2:
        return pygame.Vector2(self.rect.center)


class Arena:
    def __init__(self, arena_id: int) -> None:
        self.arena_id = arena_id
        self.bounds = pygame.Rect(PLAY_RECT)
        self.obstacles = self._create_obstacles(arena_id)
        self.destructibles = self._create_destructibles(arena_id)
        pack_positions = {
            1: (WIDTH * 0.50, HEIGHT * 0.34),
            2: (WIDTH * 0.22, HEIGHT * 0.73),
            3: (WIDTH * 0.78, HEIGHT * 0.70),
            4: (WIDTH * 0.50, HEIGHT * 0.73),
        }
        self.health_packs = [HealthPack(pygame.Vector2(pack_positions.get(arena_id, pack_positions[1])))]

    @staticmethod
    def _create_obstacles(arena_id: int) -> list[pygame.Rect]:
        if arena_id == 1:
            return [pygame.Rect(535, 260, 210, 38), pygame.Rect(535, 525, 210, 38)]
        if arena_id == 2:
            return [
                pygame.Rect(315, 215, 46, 205),
                pygame.Rect(919, 405, 46, 205),
                pygame.Rect(520, 400, 240, 42),
            ]
        if arena_id == 3:
            return [
                pygame.Rect(250, 270, 190, 40),
                pygame.Rect(840, 270, 190, 40),
                pygame.Rect(250, 525, 190, 40),
                pygame.Rect(840, 525, 190, 40),
                pygame.Rect(608, 350, 64, 130),
            ]
        return [
            pygame.Rect(355, 265, 42, 295),
            pygame.Rect(883, 265, 42, 295),
            pygame.Rect(560, 205, 160, 38),
            pygame.Rect(560, 582, 160, 38),
        ]

    @staticmethod
    def _create_destructibles(arena_id: int) -> list[DestructibleObject]:
        positions = {
            1: [(430, 390), (840, 390)],
            2: [(610, 240), (610, 545), (805, 375)],
            3: [(480, 390), (760, 390)],
            4: [(500, 300), (740, 500)],
        }
        return [DestructibleObject(pygame.Rect(x - 24, y - 24, 48, 48)) for x, y in positions.get(arena_id, [])]

    @property
    def collision_rects(self) -> list[pygame.Rect]:
        return [*self.obstacles, *(item.rect for item in self.destructibles if item.health > 0)]

    def hit_destructible(self, start: pygame.Vector2, end: pygame.Vector2, damage: float) -> pygame.Vector2 | None:
        for item in list(self.destructibles):
            if item.health > 0 and item.rect.clipline(start, end):
                item.health = max(0.0, item.health - damage)
                position = item.position
                if item.health <= 0:
                    self.destructibles.remove(item)
                return position
        return None

    def damage_destructible_near(self, position: pygame.Vector2, radius: float, damage: float) -> pygame.Vector2 | None:
        candidates = [item for item in self.destructibles if item.health > 0 and item.position.distance_to(position) <= radius]
        if not candidates:
            return None
        item = min(candidates, key=lambda candidate: candidate.position.distance_squared_to(position))
        item.health = max(0.0, item.health - damage)
        impact = item.position
        if item.health <= 0:
            self.destructibles.remove(item)
        return impact

    def reset_pickups(self) -> None:
        for pack in self.health_packs:
            pack.active = True

    @classmethod
    def from_layout(cls, layout) -> "Arena":
        arena = cls(5)
        arena.obstacles = [pygame.Rect(item) for item in layout.obstacles]
        arena.health_packs = [HealthPack(pygame.Vector2(item)) for item in layout.health_packs]
        arena.destructibles = [DestructibleObject(pygame.Rect(item)) for item in getattr(layout, "destructibles", [])]
        return arena

    def move_circle(self, position: pygame.Vector2, delta: pygame.Vector2, radius: float) -> pygame.Vector2:
        result = position.copy()
        result.x += delta.x
        result.x = max(self.bounds.left + radius, min(self.bounds.right - radius, result.x))
        result = self._resolve_axis(result, radius, horizontal=True, direction=delta.x)

        result.y += delta.y
        result.y = max(self.bounds.top + radius, min(self.bounds.bottom - radius, result.y))
        result = self._resolve_axis(result, radius, horizontal=False, direction=delta.y)
        return result

    def _resolve_axis(
        self,
        position: pygame.Vector2,
        radius: float,
        horizontal: bool,
        direction: float,
    ) -> pygame.Vector2:
        circle_rect = pygame.Rect(position.x - radius, position.y - radius, radius * 2, radius * 2)
        for obstacle in self.collision_rects:
            if not circle_rect.colliderect(obstacle):
                continue
            if horizontal:
                position.x = obstacle.left - radius if direction > 0 else obstacle.right + radius
                circle_rect.x = position.x - radius
            else:
                position.y = obstacle.top - radius if direction > 0 else obstacle.bottom + radius
                circle_rect.y = position.y - radius
        return position

    def has_line_of_sight(self, start: pygame.Vector2, end: pygame.Vector2) -> bool:
        return not any(obstacle.clipline(start, end) for obstacle in self.collision_rects)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(COLORS["background"])
        grid_color = COLORS["grid"]
        for x in range(self.bounds.left, self.bounds.right + 1, 40):
            logical_draw.line(surface, grid_color, (x, self.bounds.top), (x, self.bounds.bottom), 1)
        for y in range(self.bounds.top, self.bounds.bottom + 1, 40):
            logical_draw.line(surface, grid_color, (self.bounds.left, y), (self.bounds.right, y), 1)
        logical_draw.rect(surface, COLORS["accent"], self.bounds, 2, border_radius=8)
        for obstacle in self.obstacles:
            logical_draw.rect(surface, (27, 39, 62), obstacle, border_radius=7)
            logical_draw.rect(surface, (67, 89, 123), obstacle, 2, border_radius=7)
        for item in self.destructibles:
            ratio = item.health / item.maximum_health
            logical_draw.rect(surface, (55, 45, 31), item.rect, border_radius=5)
            logical_draw.rect(surface, COLORS["warning"], item.rect, 2, border_radius=5)
            logical_draw.line(surface, (118, 82, 40), item.rect.topleft, item.rect.bottomright, 2)
            logical_draw.line(surface, (118, 82, 40), item.rect.topright, item.rect.bottomleft, 2)
            health_rect = pygame.Rect(item.rect.left, item.rect.top - 7, item.rect.width, 3)
            logical_draw.rect(surface, (48, 35, 33), health_rect)
            logical_draw.rect(surface, COLORS["warning"], (health_rect.x, health_rect.y, int(health_rect.width * ratio), health_rect.height))
        for pack in self.health_packs:
            if not pack.active:
                continue
            center = (round(pack.position.x), round(pack.position.y))
            logical_draw.circle(surface, (24, 68, 53), center, 17)
            logical_draw.rect(surface, COLORS["health"], (center[0] - 4, center[1] - 11, 8, 22), border_radius=2)
            logical_draw.rect(surface, COLORS["health"], (center[0] - 11, center[1] - 4, 22, 8), border_radius=2)
