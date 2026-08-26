from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path

from .actions import PlayerAction


@dataclass
class SessionSummary:
    session_id: str
    player_id: str
    sample_count: int
    class_counts: dict[str, int]
    path: Path
    heatmap: list[list[int]]
    health_series: list[float]

    @property
    def favorite_action(self) -> str:
        return max(self.class_counts, key=self.class_counts.get) if self.class_counts else "IDLE"

    def rate(self, action: PlayerAction) -> float:
        total = max(1, sum(self.class_counts.values()))
        return self.class_counts.get(action.name, 0) / total


def load_sessions(directory: Path) -> list[SessionSummary]:
    sessions: list[SessionSummary] = []
    for path in directory.glob("session_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            grid = [[0 for _ in range(12)] for _ in range(6)]
            health_series: list[float] = []
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                with csv_path.open(encoding="utf-8") as file:
                    for row in csv.DictReader(file):
                        x = max(0.0, min(1.0, float(row.get("wall_distance_left", 0))))
                        y = max(0.0, min(1.0, float(row.get("wall_distance_top", 0))))
                        grid[min(5, int(y * 6))][min(11, int(x * 12))] += 1
                        health_series.append(float(row.get("player_health", 0)))
            sessions.append(
                SessionSummary(
                    str(data.get("session_id", path.stem)),
                    str(data.get("player_id", f"legacy_{data.get('session_id', path.stem)}")),
                    int(data.get("sample_count", 0)),
                    {str(key): int(value) for key, value in data.get("class_counts", {}).items()},
                    path,
                    grid,
                    health_series,
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(sessions, key=lambda item: item.path.stat().st_mtime, reverse=True)
