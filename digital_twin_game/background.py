from __future__ import annotations

from dataclasses import dataclass
import math
import random

import pygame

from .config import COLORS, HEIGHT, WIDTH


@dataclass
class Constellation:
    center: pygame.Vector2
    velocity: pygame.Vector2
    nodes: list[pygame.Vector2]
    edges: list[tuple[int, int]]
    angle: float
    rotation_speed: float
    scale: float
    phase: float


class AnimatedConstellationBackground:
    """Low-cost animated background shared by non-combat screens."""

    def __init__(self, seed: int = 731) -> None:
        generator = random.Random(seed)
        self.time = 0.0
        self.constellations: list[Constellation] = []
        for _ in range(8):
            node_count = generator.randint(4, 7)
            nodes = [
                pygame.Vector2(generator.uniform(-75, 75), generator.uniform(-55, 55))
                for _ in range(node_count)
            ]
            edges = [(index, index + 1) for index in range(node_count - 1)]
            if node_count >= 5:
                edges.append((0, generator.randint(2, node_count - 1)))
            self.constellations.append(
                Constellation(
                    center=pygame.Vector2(generator.uniform(0, WIDTH), generator.uniform(0, HEIGHT)),
                    velocity=pygame.Vector2(generator.uniform(-13, 13), generator.uniform(-7, 7)),
                    nodes=nodes,
                    edges=edges,
                    angle=generator.uniform(0, math.tau),
                    rotation_speed=generator.uniform(-0.075, 0.075),
                    scale=generator.uniform(0.75, 1.35),
                    phase=generator.uniform(0, math.tau),
                )
            )
        self.stars = [
            (
                pygame.Vector2(generator.uniform(0, WIDTH), generator.uniform(0, HEIGHT)),
                generator.uniform(0.8, 2.0),
                generator.uniform(0, math.tau),
            )
            for _ in range(42)
        ]

    def update(self, dt: float) -> None:
        self.time += dt
        margin = 130
        for constellation in self.constellations:
            constellation.center += constellation.velocity * dt
            constellation.angle += constellation.rotation_speed * dt
            if constellation.center.x < -margin:
                constellation.center.x = WIDTH + margin
            elif constellation.center.x > WIDTH + margin:
                constellation.center.x = -margin
            if constellation.center.y < -margin:
                constellation.center.y = HEIGHT + margin
            elif constellation.center.y > HEIGHT + margin:
                constellation.center.y = -margin

    def draw(self, surface: pygame.Surface, intensity: float = 1.0) -> None:
        surface.fill(COLORS["background"])
        # Wide moving bands give the flat background some depth without alpha layers.
        for index in range(5):
            y = int((index * 190 + self.time * (7 + index)) % (HEIGHT + 190) - 95)
            shade = 9 + index * 2
            pygame.draw.rect(surface, (7, shade, shade + 12), (0, y, WIDTH, 74))

        for position, radius, phase in self.stars:
            brightness = 0.72 + 0.18 * math.sin(phase)
            value = int((35 + 55 * brightness) * intensity)
            pygame.draw.circle(surface, (max(10, value // 3), value, value + 18), position, max(1, int(radius)))

        for constellation in self.constellations:
            brightness = (0.76 + 0.14 * math.sin(constellation.phase)) * intensity
            cosine = math.cos(constellation.angle)
            sine = math.sin(constellation.angle)
            points: list[pygame.Vector2] = []
            for node in constellation.nodes:
                x = (node.x * cosine - node.y * sine) * constellation.scale
                y = (node.x * sine + node.y * cosine) * constellation.scale
                points.append(constellation.center + pygame.Vector2(x, y))
            line_color = (
                int(15 + 20 * brightness),
                int(48 + 55 * brightness),
                int(62 + 70 * brightness),
            )
            node_color = (
                int(20 + 20 * brightness),
                int(90 + 75 * brightness),
                int(105 + 80 * brightness),
            )
            for start, end in constellation.edges:
                pygame.draw.aaline(surface, line_color, points[start], points[end])
            for index, point in enumerate(points):
                radius = 2 + int((math.sin(self.time * 2.1 + constellation.phase + index) + 1) * 0.7)
                pygame.draw.circle(surface, node_color, point, radius)
