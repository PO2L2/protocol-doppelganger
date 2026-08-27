from __future__ import annotations

from functools import lru_cache

import pygame

from .config import COLORS
from .hidpi import draw as logical_draw, logical_rect, scale_point, scale_rect, surface_scale


@lru_cache(maxsize=96)
def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont("segoeui", size, bold=bold)


def draw_text(
    surface: pygame.Surface,
    text: str,
    position: tuple[float, float],
    size: int = 24,
    color: tuple[int, int, int] | None = None,
    bold: bool = False,
    anchor: str = "topleft",
) -> pygame.Rect:
    scale = surface_scale(surface)
    image = get_font(max(1, round(size * scale)), bold).render(text, True, color or COLORS["text"])
    rect = image.get_rect()
    setattr(rect, anchor, scale_point(surface, position))
    surface.blit(image, rect)
    return logical_rect(surface, rect)


def draw_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    maximum: float,
    color: tuple[int, int, int],
    label: str = "",
) -> None:
    logical_draw.rect(surface, (26, 34, 52), rect, border_radius=rect.height // 2)
    ratio = 0.0 if maximum <= 0 else max(0.0, min(1.0, value / maximum))
    fill = rect.copy()
    fill.width = int(rect.width * ratio)
    if fill.width:
        logical_draw.rect(surface, color, fill, border_radius=rect.height // 2)
    logical_draw.rect(surface, (72, 91, 118), rect, 1, border_radius=rect.height // 2)
    if label:
        draw_text(surface, label, (rect.left, rect.top - 22), 15, COLORS["muted"])


def draw_panel(surface: pygame.Surface, rect: pygame.Rect, alpha: int = 245) -> None:
    physical_rect = scale_rect(surface, rect)
    layer = pygame.Surface(physical_rect.size, pygame.SRCALPHA)
    layer.fill((*COLORS["panel"], alpha))
    surface.blit(layer, physical_rect)
    logical_draw.rect(surface, (56, 75, 105), rect, 1, border_radius=12)


def draw_metric(
    surface: pygame.Surface,
    label: str,
    value: float,
    rect: pygame.Rect,
    color: tuple[int, int, int],
) -> None:
    draw_text(surface, label, (rect.left, rect.top), 18, COLORS["muted"])
    bar = pygame.Rect(rect.left, rect.top + 28, rect.width, 12)
    draw_bar(surface, bar, value, 1.0, color)
    draw_text(surface, f"{value * 100:.0f}%", (rect.right, rect.top), 18, COLORS["text"], bold=True, anchor="topright")
