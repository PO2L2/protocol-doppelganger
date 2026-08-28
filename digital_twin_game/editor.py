from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import pygame

from .hidpi import draw as logical_draw

from .config import COLORS, PLAY_RECT
from .weapons import WEAPON_SPECS, WeaponType


EDITOR_ENEMIES = ["assault", "sniper", "shield", "teleporter", "engineer", "copier", "twin"]
EDITOR_MODES = ["place", "move", "erase"]
GAME_MODES = ["elimination", "survival", "hold"]
GAME_MODE_NAMES = {
    "elimination": "Уничтожение",
    "survival": "Выживание 45 секунд",
    "hold": "Удержание точки",
}
EDIT_MODE_NAMES = {"place": "Размещение", "move": "Перемещение", "erase": "Удаление"}
WALL_LENGTHS = [80, 160, 240]


@dataclass
class ArenaLayout:
    obstacles: list[tuple[int, int, int, int]] = field(default_factory=list)
    health_packs: list[tuple[int, int]] = field(default_factory=list)
    player_spawn: tuple[int, int] = (140, 400)
    enemy_spawns: list[tuple[int, int, str]] = field(default_factory=list)
    game_mode: str = "elimination"
    objective_position: tuple[int, int] = (640, 400)
    destructibles: list[tuple[int, int, int, int]] = field(default_factory=list)
    player_weapon: str = WeaponType.PULSE.value

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ArenaLayout":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                obstacles=[tuple(item) for item in data.get("obstacles", [])],
                health_packs=[tuple(item) for item in data.get("health_packs", [])],
                player_spawn=tuple(data.get("player_spawn", (140, 400))),
                enemy_spawns=[tuple(item) for item in data.get("enemy_spawns", [])],
                game_mode=data.get("game_mode", "elimination") if data.get("game_mode", "elimination") in GAME_MODES else "elimination",
                objective_position=tuple(data.get("objective_position", (640, 400))),
                destructibles=[tuple(item) for item in data.get("destructibles", [])],
                player_weapon=(
                    data.get("player_weapon", WeaponType.PULSE.value)
                    if data.get("player_weapon", WeaponType.PULSE.value) in {weapon.value for weapon in WeaponType}
                    else WeaponType.PULSE.value
                ),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return cls()


class ArenaEditor:
    def __init__(self, layout: ArenaLayout | None = None) -> None:
        self.bounds = pygame.Rect(PLAY_RECT)
        self.layout = layout or ArenaLayout()
        self.tool = 1
        self.enemy_index = 0
        self.message = ""
        self.edit_mode_index = 0
        self.wall_length_index = 0
        self.wall_vertical = False
        self.selected_obstacle: int | None = None
        self.dragging = False
        self.weapon_button_rect = pygame.Rect(1000, 14, 232, 50)

    @property
    def enemy_kind(self) -> str:
        return EDITOR_ENEMIES[self.enemy_index]

    @property
    def edit_mode(self) -> str:
        return EDITOR_MODES[self.edit_mode_index]

    @property
    def wall_size(self) -> tuple[int, int]:
        length = WALL_LENGTHS[self.wall_length_index]
        return (40, length) if self.wall_vertical else (length, 40)

    @property
    def player_weapon(self) -> WeaponType:
        try:
            return WeaponType(self.layout.player_weapon)
        except ValueError:
            self.layout.player_weapon = WeaponType.PULSE.value
            return WeaponType.PULSE

    def cycle_weapon(self) -> None:
        weapons = list(WeaponType)
        index = weapons.index(self.player_weapon)
        selected = weapons[(index + 1) % len(weapons)]
        self.layout.player_weapon = selected.value
        self.message = f"Оружие игрока: {WEAPON_SPECS[selected].name}"

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key <= pygame.K_6:
                self.tool = event.key - pygame.K_0
            elif event.key == pygame.K_TAB:
                self.enemy_index = (self.enemy_index + 1) % len(EDITOR_ENEMIES)
            elif event.key == pygame.K_m:
                self.edit_mode_index = (self.edit_mode_index + 1) % len(EDITOR_MODES)
                self.dragging = False
            elif event.key == pygame.K_g:
                current = GAME_MODES.index(self.layout.game_mode) if self.layout.game_mode in GAME_MODES else 0
                self.layout.game_mode = GAME_MODES[(current + 1) % len(GAME_MODES)]
                self.message = f"Режим арены: {GAME_MODE_NAMES[self.layout.game_mode]}"
            elif event.key == pygame.K_c:
                self.wall_length_index = (self.wall_length_index + 1) % len(WALL_LENGTHS)
            elif event.key == pygame.K_w:
                self.cycle_weapon()
            elif event.key == pygame.K_r:
                if self.selected_obstacle is not None:
                    self.rotate_selected_wall()
                else:
                    self.wall_vertical = not self.wall_vertical
        elif event.type == pygame.MOUSEMOTION and self.dragging and self.edit_mode == "move":
            self._move_selected(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.weapon_button_rect.collidepoint(event.pos):
            self.cycle_weapon()
        elif event.type == pygame.MOUSEBUTTONDOWN and self.bounds.collidepoint(event.pos):
            if self.edit_mode == "erase":
                self._remove(event.pos)
            elif self.edit_mode == "move" and event.button == 1:
                self.selected_obstacle = self._obstacle_at(event.pos)
                self.dragging = self.selected_obstacle is not None
            elif event.button == 1:
                self._place(event.pos)
            elif event.button == 3:
                self._remove(event.pos)

    def _snap(self, position: tuple[int, int], grid: int = 40) -> tuple[int, int]:
        return round(position[0] / grid) * grid, round(position[1] / grid) * grid

    def _clamp_point(self, position: tuple[int, int], margin: int) -> tuple[int, int]:
        return (
            max(self.bounds.left + margin, min(self.bounds.right - margin, position[0])),
            max(self.bounds.top + margin, min(self.bounds.bottom - margin, position[1])),
        )

    def _wall_rect(self, position: tuple[int, int]) -> pygame.Rect:
        preview_x, preview_y = self._snap(position)
        preview_width, preview_height = self.wall_size
        preview = pygame.Rect(0, 0, preview_width, preview_height)
        preview.center = (preview_x, preview_y)
        preview.clamp_ip(self.bounds)
        return preview

    def _wall_is_free(self, rect: pygame.Rect) -> bool:
        occupied = [pygame.Rect(item) for item in self.layout.obstacles + self.layout.destructibles]
        return not any(rect.colliderect(item) for item in occupied)

    def _nearest_free_wall_rect(self, position: tuple[int, int]) -> pygame.Rect | None:
        desired = self._wall_rect(position)
        if self._wall_is_free(desired):
            return desired

        width, height = desired.size
        primary_candidates: list[pygame.Rect] = []
        secondary_candidates: list[pygame.Rect] = []
        for item in self.layout.obstacles + self.layout.destructibles:
            obstacle = pygame.Rect(item)
            if self.wall_vertical:
                primary_candidates.extend(
                    [pygame.Rect(desired.left, obstacle.bottom, width, height), pygame.Rect(desired.left, obstacle.top - height, width, height)]
                )
                secondary_candidates.extend(
                    [pygame.Rect(obstacle.right, desired.top, width, height), pygame.Rect(obstacle.left - width, desired.top, width, height)]
                )
            else:
                primary_candidates.extend(
                    [pygame.Rect(obstacle.right, desired.top, width, height), pygame.Rect(obstacle.left - width, desired.top, width, height)]
                )
                secondary_candidates.extend(
                    [pygame.Rect(desired.left, obstacle.bottom, width, height), pygame.Rect(desired.left, obstacle.top - height, width, height)]
                )

        for candidates in (primary_candidates, secondary_candidates):
            valid = [candidate for candidate in candidates if self.bounds.contains(candidate) and self._wall_is_free(candidate)]
            if valid:
                return min(valid, key=lambda candidate: pygame.Vector2(candidate.center).distance_squared_to(desired.center))
        return None

    def wall_preview_rect(self, position: tuple[int, int]) -> pygame.Rect:
        return self._nearest_free_wall_rect(position) or self._wall_rect(position)

    def _place(self, position: tuple[int, int]) -> None:
        x, y = self._snap(position)
        if self.tool == 1:
            rect = self._nearest_free_wall_rect(position)
            if rect is None:
                self.message = "Рядом нет свободного места для стены"
                return
            self.layout.obstacles.append(tuple(rect))
            self.selected_obstacle = len(self.layout.obstacles) - 1
        elif self.tool == 2:
            x, y = self._clamp_point((x, y), 18)
            self.layout.health_packs.append((x, y))
        elif self.tool == 3:
            x, y = self._clamp_point((x, y), 20)
            self.layout.player_spawn = (x, y)
        elif self.tool == 4:
            x, y = self._clamp_point((x, y), 22)
            self.layout.enemy_spawns.append((x, y, self.enemy_kind))
        elif self.tool == 5:
            self.layout.objective_position = self._clamp_point((x, y), 40)
        elif self.tool == 6:
            crate = pygame.Rect(x - 24, y - 24, 48, 48).clamp(self.bounds)
            occupied = [pygame.Rect(item) for item in self.layout.obstacles + self.layout.destructibles]
            if not any(crate.colliderect(item) for item in occupied):
                self.layout.destructibles.append(tuple(crate))

    def _obstacle_at(self, position: tuple[int, int]) -> int | None:
        for index in range(len(self.layout.obstacles) - 1, -1, -1):
            if pygame.Rect(self.layout.obstacles[index]).inflate(12, 12).collidepoint(position):
                return index
        return None

    def _move_selected(self, position: tuple[int, int]) -> None:
        if self.selected_obstacle is None or self.selected_obstacle >= len(self.layout.obstacles):
            return
        x, y = self._snap(position)
        old = pygame.Rect(self.layout.obstacles[self.selected_obstacle])
        moved = pygame.Rect(0, 0, old.width, old.height)
        moved.center = (x, y)
        moved.clamp_ip(self.bounds)
        if any(moved.colliderect(pygame.Rect(item)) for index, item in enumerate(self.layout.obstacles) if index != self.selected_obstacle):
            return
        self.layout.obstacles[self.selected_obstacle] = tuple(moved)

    def rotate_selected_wall(self) -> bool:
        if self.selected_obstacle is None or self.selected_obstacle >= len(self.layout.obstacles):
            return False
        old = pygame.Rect(self.layout.obstacles[self.selected_obstacle])
        rotated = pygame.Rect(0, 0, old.height, old.width)
        rotated.center = old.center
        rotated.clamp_ip(self.bounds)
        if any(rotated.colliderect(pygame.Rect(item)) for index, item in enumerate(self.layout.obstacles) if index != self.selected_obstacle):
            self.message = "Повороту мешает другая стена"
            return False
        self.layout.obstacles[self.selected_obstacle] = tuple(rotated)
        self.wall_vertical = rotated.height > rotated.width
        self.message = "Стена повёрнута"
        return True

    def _remove(self, position: tuple[int, int]) -> None:
        point = pygame.Vector2(position)
        for obstacle in reversed(self.layout.obstacles):
            if pygame.Rect(obstacle).inflate(10, 10).collidepoint(position):
                index = self.layout.obstacles.index(obstacle)
                self.layout.obstacles.remove(obstacle)
                if self.selected_obstacle == index:
                    self.selected_obstacle = None
                elif self.selected_obstacle is not None and self.selected_obstacle > index:
                    self.selected_obstacle -= 1
                return
        if self.layout.health_packs:
            nearest = min(self.layout.health_packs, key=lambda item: point.distance_squared_to(item))
            if point.distance_to(nearest) < 35:
                self.layout.health_packs.remove(nearest)
                return
        for destructible in reversed(self.layout.destructibles):
            if pygame.Rect(destructible).inflate(10, 10).collidepoint(position):
                self.layout.destructibles.remove(destructible)
                return
        if self.layout.enemy_spawns:
            nearest_enemy = min(self.layout.enemy_spawns, key=lambda item: point.distance_squared_to(item[:2]))
            if point.distance_to(nearest_enemy[:2]) < 35:
                self.layout.enemy_spawns.remove(nearest_enemy)

    def draw(self, surface: pygame.Surface, draw_text, draw_background=None, mouse_position=None) -> None:
        if draw_background is None:
            surface.fill(COLORS["background"])
        else:
            draw_background(surface, 0.52)
        mouse_position = pygame.mouse.get_pos() if mouse_position is None else mouse_position
        animation_time = pygame.time.get_ticks() / 1000.0
        grid_color = tuple(int(channel * 0.76) for channel in COLORS["grid"])
        for x in range(self.bounds.left, self.bounds.right + 1, 40):
            logical_draw.line(surface, grid_color, (x, self.bounds.top), (x, self.bounds.bottom))
        for y in range(self.bounds.top, self.bounds.bottom + 1, 40):
            logical_draw.line(surface, grid_color, (self.bounds.left, y), (self.bounds.right, y))
        scan_y = self.bounds.top + int((animation_time * 72) % self.bounds.height)
        logical_draw.line(surface, (20, 72, 89), (self.bounds.left, scan_y), (self.bounds.right, scan_y), 2)
        logical_draw.line(surface, (12, 39, 55), (self.bounds.left, scan_y + 5), (self.bounds.right, scan_y + 5), 1)
        logical_draw.rect(surface, COLORS["accent"], self.bounds, 2)
        for index, obstacle in enumerate(self.layout.obstacles):
            logical_draw.rect(surface, (35, 52, 80), obstacle, border_radius=5)
            outline = COLORS["warning"] if index == self.selected_obstacle else (77, 100, 139)
            logical_draw.rect(surface, outline, obstacle, 3 if index == self.selected_obstacle else 2, border_radius=5)
            if index == self.selected_obstacle:
                glow = 5 + int((math.sin(animation_time * 4.5) + 1) * 2)
                logical_draw.rect(surface, (112, 77, 24), pygame.Rect(obstacle).inflate(glow, glow), 1, border_radius=7)
        for position in self.layout.health_packs:
            logical_draw.circle(surface, COLORS["health"], position, 12)
            draw_text(surface, "+", position, 20, COLORS["white"], bold=True, anchor="center")
        for destructible in self.layout.destructibles:
            rect = pygame.Rect(destructible)
            logical_draw.rect(surface, (55, 45, 31), rect, border_radius=5)
            logical_draw.rect(surface, COLORS["warning"], rect, 2, border_radius=5)
            logical_draw.line(surface, (118, 82, 40), rect.topleft, rect.bottomright, 2)
            logical_draw.line(surface, (118, 82, 40), rect.topright, rect.bottomleft, 2)
        logical_draw.circle(surface, COLORS["player"], self.layout.player_spawn, 18)
        for x, y, kind in self.layout.enemy_spawns:
            logical_draw.circle(surface, COLORS["enemy"], (x, y), 18)
            draw_text(surface, kind[:2].upper(), (x, y), 11, COLORS["white"], bold=True, anchor="center")
        objective = self.layout.objective_position
        objective_color = COLORS["warning"] if self.layout.game_mode == "hold" else COLORS["muted"]
        objective_radius = 36 + int((math.sin(animation_time * 3.2) + 1) * 3)
        logical_draw.circle(surface, objective_color, objective, objective_radius, 2)
        logical_draw.line(surface, objective_color, (objective[0] - 8, objective[1]), (objective[0] + 8, objective[1]), 2)
        logical_draw.line(surface, objective_color, (objective[0], objective[1] - 8), (objective[0], objective[1] + 8), 2)
        if self.edit_mode == "place" and self.tool == 1 and self.bounds.collidepoint(mouse_position):
            preview = self.wall_preview_rect(mouse_position)
            logical_draw.rect(surface, (48, 75, 91), preview, 2, border_radius=5)
        draw_text(surface, "РЕДАКТОР АРЕН", (30, 18), 26, COLORS["player"], bold=True)
        tools = "1 Стена  2 Аптечка  3 Игрок  4 Враг  5 Цель  6 Контейнер  TAB Вид врага"
        draw_text(surface, tools, (290, 16), 15, COLORS["text"])
        draw_text(surface, "M Редактирование  R Поворот  C Размер  G Режим арены  W Оружие", (290, 42), 14, COLORS["muted"])
        weapon_hovered = self.weapon_button_rect.collidepoint(mouse_position)
        logical_draw.rect(surface, (24, 61, 70) if weapon_hovered else COLORS["panel"], self.weapon_button_rect, border_radius=8)
        logical_draw.rect(surface, COLORS["player"] if weapon_hovered else (61, 79, 108), self.weapon_button_rect, 2, border_radius=8)
        draw_text(
            surface,
            f"W • {WEAPON_SPECS[self.player_weapon].name}",
            self.weapon_button_rect.center,
            13,
            COLORS["white"],
            bold=True,
            anchor="center",
        )
        tool_names = {1: "Стена", 2: "Аптечка", 3: "Точка игрока", 4: f"Враг: {self.enemy_kind}", 5: "Точка цели", 6: "Разрушаемый контейнер"}
        wall_info = f" • {self.wall_size[0]}×{self.wall_size[1]}" if self.tool == 1 else ""
        draw_text(surface, f"{EDIT_MODE_NAMES[self.edit_mode]} • {tool_names[self.tool]}{wall_info}", (30, 682), 15, COLORS["warning"])
        draw_text(surface, f"Арена: {GAME_MODE_NAMES[self.layout.game_mode]}", (520, 682), 15, COLORS["player"], bold=True)
        draw_text(surface, "ЛКМ действие • ПКМ удалить • S сохранить • T проверить • ESC меню", (1250, 682), 14, COLORS["muted"], anchor="topright")
        if self.message:
            draw_text(surface, self.message, (self.bounds.centerx, self.bounds.top + 20), 18, COLORS["health"], bold=True, anchor="midtop")
