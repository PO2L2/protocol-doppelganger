"""Aspect-preserving presentation of the fixed logical game canvas."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


# Resolutions explicitly requested for the Windows build.  The viewport also
# accepts arbitrary display sizes, so this list doubles as a regression suite.
SUPPORTED_DISPLAY_SIZES = (
    (3620, 2036),
    (2560, 1440),
    (1920, 1440),
    (1920, 1200),
    (1920, 1080),
    (1680, 1050),
    (1600, 1200),
    (1600, 1024),
    (1600, 900),
    (1440, 1080),
    (1440, 900),
    (1366, 768),
    (1360, 768),
    (1280, 1024),
    (1280, 960),
    (1280, 800),
    (1280, 768),
    (1280, 720),
    (1176, 664),
    (1152, 864),
    (1024, 768),
    (800, 600),
)


@dataclass
class DisplayViewport:
    logical_size: tuple[int, int]
    display_size: tuple[int, int]

    def __post_init__(self) -> None:
        self.update(self.display_size)

    def update(self, display_size: tuple[int, int]) -> None:
        display_width = max(1, int(display_size[0]))
        display_height = max(1, int(display_size[1]))
        logical_width, logical_height = self.logical_size
        scale = min(display_width / logical_width, display_height / logical_height)
        width = max(1, round(logical_width * scale))
        height = max(1, round(logical_height * scale))
        self.display_size = (display_width, display_height)
        self.scale = scale
        self.rect = pygame.Rect((display_width - width) // 2, (display_height - height) // 2, width, height)

    def display_to_logical(self, position: tuple[int, int], *, clamp: bool = False) -> tuple[int, int]:
        x, y = position
        if not self.rect.collidepoint(x, y):
            if not clamp:
                return (-10_000, -10_000)
            x = min(max(x, self.rect.left), self.rect.right - 1)
            y = min(max(y, self.rect.top), self.rect.bottom - 1)
        logical_width, logical_height = self.logical_size
        logical_x = int((x - self.rect.left) * logical_width / self.rect.width)
        logical_y = int((y - self.rect.top) * logical_height / self.rect.height)
        return (
            min(logical_width - 1, max(0, logical_x)),
            min(logical_height - 1, max(0, logical_y)),
        )

    def logical_to_display(self, position: tuple[float, float]) -> tuple[int, int]:
        logical_width, logical_height = self.logical_size
        return (
            self.rect.left + round(position[0] * self.rect.width / logical_width),
            self.rect.top + round(position[1] * self.rect.height / logical_height),
        )
