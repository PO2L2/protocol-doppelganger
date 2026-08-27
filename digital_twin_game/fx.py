from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import random

import pygame

from .hidpi import draw as logical_draw, scale_point, surface_scale
from .ui import get_font


MAX_PARTICLES = 420
MAX_FLOATING_TEXTS = 36


@dataclass
class Particle:
    position: pygame.Vector2
    velocity: pygame.Vector2
    color: tuple[int, int, int]
    lifetime: float
    maximum_lifetime: float
    radius: float

    def update(self, dt: float) -> bool:
        self.position += self.velocity * dt
        self.velocity *= max(0.0, 1.0 - dt * 3.0)
        self.lifetime -= dt
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface, offset: pygame.Vector2) -> None:
        ratio = max(0.0, self.lifetime / self.maximum_lifetime)
        # A full-screen alpha surface per particle was the main FPS bottleneck.
        # Fading the RGB value preserves the look without large allocations.
        brightness = 0.22 + ratio * 0.78
        faded_color = tuple(max(0, min(255, int(channel * brightness))) for channel in self.color)
        logical_draw.circle(surface, faded_color, self.position + offset, max(1, int(self.radius * ratio)))


@dataclass
class FloatingText:
    text: str
    position: pygame.Vector2
    color: tuple[int, int, int]
    lifetime: float = 0.8

    def update(self, dt: float) -> bool:
        self.position.y -= 38 * dt
        self.lifetime -= dt
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface, offset: pygame.Vector2) -> None:
        scale = surface_scale(surface)
        font = get_font(max(1, round(17 * scale)), True)
        image = font.render(self.text, True, self.color)
        image.set_alpha(max(0, min(255, int(self.lifetime / 0.8 * 255))))
        physical_position = scale_point(surface, self.position + offset)
        surface.blit(image, image.get_rect(center=physical_position))


class CombatEffects:
    def __init__(self) -> None:
        self.particles: list[Particle] = []
        self.texts: list[FloatingText] = []
        self.shake_timer = 0.0
        self.shake_strength = 0.0

    def update(self, dt: float) -> None:
        self.particles = [particle for particle in self.particles if particle.update(dt)]
        self.texts = [text for text in self.texts if text.update(dt)]
        self.shake_timer = max(0.0, self.shake_timer - dt)
        if self.shake_timer <= 0:
            self.shake_strength = 0.0

    def _limit_effects(self) -> None:
        if len(self.particles) > MAX_PARTICLES:
            del self.particles[:-MAX_PARTICLES]
        if len(self.texts) > MAX_FLOATING_TEXTS:
            del self.texts[:-MAX_FLOATING_TEXTS]

    def hit(self, position: pygame.Vector2, color: tuple[int, int, int], damage: float, strong: bool = False) -> None:
        count = 18 if strong else 10
        for _ in range(count):
            angle = random.random() * math.tau
            speed = random.uniform(70, 230 if strong else 160)
            self.particles.append(
                Particle(position.copy(), pygame.Vector2(math.cos(angle), math.sin(angle)) * speed, color, 0.45, 0.45, random.uniform(2, 5))
            )
        self.texts.append(FloatingText(f"-{damage:.0f}", position.copy(), color))
        self._limit_effects()
        self.shake_timer = 0.18 if strong else 0.09
        self.shake_strength = 9 if strong else 4

    def dash(self, position: pygame.Vector2, color: tuple[int, int, int], direction: pygame.Vector2) -> None:
        for _ in range(16):
            jitter = pygame.Vector2(random.uniform(-25, 25), random.uniform(-25, 25))
            self.particles.append(Particle(position.copy() + jitter, -direction * random.uniform(40, 140), color, 0.35, 0.35, random.uniform(3, 7)))
        self._limit_effects()

    def wave(self, position: pygame.Vector2, color: tuple[int, int, int]) -> None:
        for index in range(36):
            angle = index / 36 * math.tau
            self.particles.append(Particle(position.copy(), pygame.Vector2(math.cos(angle), math.sin(angle)) * 210, color, 0.55, 0.55, 4))
        self._limit_effects()
        self.shake_timer = 0.22
        self.shake_strength = 7

    def offset(self) -> pygame.Vector2:
        if self.shake_timer <= 0:
            return pygame.Vector2()
        return pygame.Vector2(random.uniform(-self.shake_strength, self.shake_strength), random.uniform(-self.shake_strength, self.shake_strength))

    def draw(self, surface: pygame.Surface, offset: pygame.Vector2) -> None:
        for particle in self.particles:
            particle.draw(surface, offset)
        for text in self.texts:
            text.draw(surface, offset)


class AudioManager:
    """Small procedural sound set; safely disables itself if audio is unavailable."""

    def __init__(self) -> None:
        self.enabled = False
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        self.ambient: pygame.mixer.Sound | None = None
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=22050, size=-16, channels=1)
            self.sounds = {
                "shot": self._tone(520, 0.07, 0.16),
                "hit": self._tone(120, 0.10, 0.20),
                "dash": self._tone(760, 0.06, 0.12),
                "combo": self._tone(980, 0.18, 0.14),
                "perfect": self._tone(1250, 0.24, 0.15),
                "ability": self._tone(680, 0.20, 0.14),
            }
            self.ambient = self._ambient_loop()
            self.ambient.set_volume(0.06)
            self.ambient.play(loops=-1)
            self.enabled = True
        except pygame.error:
            self.enabled = False

    @staticmethod
    def _tone(frequency: float, duration: float, volume: float) -> pygame.mixer.Sound:
        sample_rate = 22050
        count = int(sample_rate * duration)
        samples = array("h")
        for index in range(count):
            envelope = 1.0 - index / count
            value = int(32767 * volume * envelope * math.sin(math.tau * frequency * index / sample_rate))
            samples.append(value)
        return pygame.mixer.Sound(buffer=samples)

    @staticmethod
    def _ambient_loop() -> pygame.mixer.Sound:
        sample_rate = 22050
        count = sample_rate * 2
        samples = array("h")
        for index in range(count):
            time = index / sample_rate
            value = 0.035 * math.sin(math.tau * 55 * time) + 0.018 * math.sin(math.tau * 82.5 * time)
            samples.append(int(32767 * value))
        return pygame.mixer.Sound(buffer=samples)

    def play(self, name: str) -> None:
        if self.enabled and name in self.sounds:
            self.sounds[name].play()
