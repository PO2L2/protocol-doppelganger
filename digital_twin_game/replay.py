from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pygame

from .actions import PlayerAction


@dataclass
class ReplayFrame:
    arena_id: int
    arena_time: float
    normalized_x: float
    normalized_y: float
    action_id: int


@dataclass
class ReplayEvent:
    arena_id: int
    arena_time: float
    name: str


class ReplayRecorder:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.timer = 0.0
        self.frames: list[ReplayFrame] = []
        self.events: list[ReplayEvent] = []

    def update(self, dt: float, arena_id: int, arena_time: float, position: pygame.Vector2, bounds: pygame.Rect, action: PlayerAction) -> None:
        self.timer += dt
        if self.timer < self.interval:
            return
        self.timer %= self.interval
        self.frames.append(
            ReplayFrame(
                arena_id,
                arena_time,
                max(0.0, min(1.0, (position.x - bounds.left) / bounds.width)),
                max(0.0, min(1.0, (position.y - bounds.top) / bounds.height)),
                int(action),
            )
        )

    def add_event(self, arena_id: int, arena_time: float, name: str) -> None:
        self.events.append(ReplayEvent(arena_id, arena_time, name))

    def frames_for_arena(self, arena_id: int) -> list[ReplayFrame]:
        return [frame for frame in self.frames if frame.arena_id == arena_id]

    def position_at(self, arena_id: int, arena_time: float, bounds: pygame.Rect) -> pygame.Vector2 | None:
        frames = self.frames_for_arena(arena_id)
        if not frames:
            return None
        frame = min(frames, key=lambda item: abs(item.arena_time - arena_time))
        return pygame.Vector2(bounds.left + frame.normalized_x * bounds.width, bounds.top + frame.normalized_y * bounds.height)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "frames": [asdict(frame) for frame in self.frames],
            "events": [asdict(event) for event in self.events],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReplayRecorder | None":
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            recorder = cls()
            recorder.frames = [ReplayFrame(**item) for item in payload.get("frames", [])]
            recorder.events = [ReplayEvent(**item) for item in payload.get("events", [])]
            return recorder
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

