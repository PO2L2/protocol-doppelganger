"""Persistent named player profiles used to keep training data separated."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PlayerProfileRecord:
    player_id: str
    name: str
    created_at: float
    last_used_at: float
    session_count: int = 0


class PlayerRegistry:
    def __init__(self, path: Path, profiles: list[PlayerProfileRecord] | None = None, active_id: str = "") -> None:
        self.path = path
        self.profiles = profiles or []
        self.active_id = active_id
        if not self.profiles:
            self.create("Игрок")
        if not any(profile.player_id == self.active_id for profile in self.profiles):
            self.active_id = self.profiles[0].player_id

    @classmethod
    def load(cls, path: Path) -> "PlayerRegistry":
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                profiles = [PlayerProfileRecord(**item) for item in payload.get("profiles", [])]
                return cls(path, profiles, str(payload.get("active_id", "")))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return cls(path)

    @property
    def active(self) -> PlayerProfileRecord:
        return next(profile for profile in self.profiles if profile.player_id == self.active_id)

    def create(self, name: str) -> PlayerProfileRecord:
        cleaned = " ".join(name.strip().split())[:24]
        if not cleaned:
            raise ValueError("Имя игрока не может быть пустым")
        existing = next((profile for profile in self.profiles if profile.name.casefold() == cleaned.casefold()), None)
        if existing is not None:
            self.select(existing.player_id)
            return existing
        now = time.time()
        profile = PlayerProfileRecord(uuid.uuid4().hex[:10], cleaned, now, now)
        self.profiles.append(profile)
        self.active_id = profile.player_id
        self.save()
        return profile

    def select(self, player_id: str) -> PlayerProfileRecord:
        profile = next(profile for profile in self.profiles if profile.player_id == player_id)
        self.active_id = player_id
        profile.last_used_at = time.time()
        self.save()
        return profile

    def record_session(self, player_id: str) -> None:
        profile = next((profile for profile in self.profiles if profile.player_id == player_id), None)
        if profile is None:
            return
        profile.session_count += 1
        profile.last_used_at = time.time()
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "active_id": self.active_id,
            "profiles": [asdict(profile) for profile in self.profiles],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
