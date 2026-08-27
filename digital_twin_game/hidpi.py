"""Logical-coordinate drawing on native-resolution pygame surfaces."""

from __future__ import annotations

from typing import Iterable

import pygame


_surface_scales: dict[pygame.Surface, float] = {}


def register_surface(surface: pygame.Surface, scale: float) -> pygame.Surface:
    _surface_scales[surface] = max(0.01, float(scale))
    return surface


def unregister_surface(surface: pygame.Surface) -> None:
    _surface_scales.pop(surface, None)


def surface_scale(surface: pygame.Surface) -> float:
    return _surface_scales.get(surface, 1.0)


def scale_point(surface: pygame.Surface, point) -> tuple[float, float]:
    scale = surface_scale(surface)
    return float(point[0]) * scale, float(point[1]) * scale


def scale_rect(surface: pygame.Surface, rect) -> pygame.Rect:
    scale = surface_scale(surface)
    logical = pygame.Rect(rect)
    return pygame.Rect(
        round(logical.x * scale),
        round(logical.y * scale),
        round(logical.width * scale),
        round(logical.height * scale),
    )


def logical_rect(surface: pygame.Surface, rect: pygame.Rect) -> pygame.Rect:
    scale = surface_scale(surface)
    return pygame.Rect(
        round(rect.x / scale),
        round(rect.y / scale),
        round(rect.width / scale),
        round(rect.height / scale),
    )


def scale_width(surface: pygame.Surface, width: int | float) -> int:
    if width <= 0:
        return 0
    return max(1, round(float(width) * surface_scale(surface)))


def blit(surface: pygame.Surface, source: pygame.Surface, destination, area=None, special_flags: int = 0) -> pygame.Rect:
    if isinstance(destination, pygame.Rect):
        physical_destination = scale_rect(surface, destination)
    else:
        physical_destination = scale_point(surface, destination)
    return surface.blit(source, physical_destination, area, special_flags)


class LogicalDraw:
    @staticmethod
    def rect(surface, color, rect, width=0, border_radius=0, border_top_left_radius=-1,
             border_top_right_radius=-1, border_bottom_left_radius=-1, border_bottom_right_radius=-1):
        scale = surface_scale(surface)

        def radius(value: int) -> int:
            return value if value < 0 else round(value * scale)

        return pygame.draw.rect(
            surface,
            color,
            scale_rect(surface, rect),
            scale_width(surface, width),
            radius(border_radius),
            radius(border_top_left_radius),
            radius(border_top_right_radius),
            radius(border_bottom_left_radius),
            radius(border_bottom_right_radius),
        )

    @staticmethod
    def circle(surface, color, center, radius, width=0, draw_top_right=True, draw_top_left=True,
               draw_bottom_left=True, draw_bottom_right=True):
        return pygame.draw.circle(
            surface,
            color,
            scale_point(surface, center),
            max(1, round(float(radius) * surface_scale(surface))),
            scale_width(surface, width),
            draw_top_right,
            draw_top_left,
            draw_bottom_left,
            draw_bottom_right,
        )

    @staticmethod
    def line(surface, color, start_pos, end_pos, width=1):
        return pygame.draw.line(
            surface,
            color,
            scale_point(surface, start_pos),
            scale_point(surface, end_pos),
            scale_width(surface, width),
        )

    @staticmethod
    def lines(surface, color, closed, points: Iterable, width=1):
        return pygame.draw.lines(
            surface,
            color,
            closed,
            [scale_point(surface, point) for point in points],
            scale_width(surface, width),
        )

    @staticmethod
    def aaline(surface, color, start_pos, end_pos, blend=1):
        return pygame.draw.aaline(
            surface,
            color,
            scale_point(surface, start_pos),
            scale_point(surface, end_pos),
            blend,
        )

    @staticmethod
    def arc(surface, color, rect, start_angle, stop_angle, width=1):
        return pygame.draw.arc(
            surface,
            color,
            scale_rect(surface, rect),
            start_angle,
            stop_angle,
            scale_width(surface, width),
        )


draw = LogicalDraw()
