from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .actions import PlayerAction
from .model_interface import FEATURE_NAMES, validate_features


@dataclass
class TrainingSample:
    timestamp: float
    session_id: str
    arena_id: int
    features: tuple[float, ...]
    action_id: int
    player_id: str = "legacy"
    arena_time: float = 0.0
    reward: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    successful_block: int = 0
    perfect_dodge: int = 0
    effective_heal: float = 0.0
    kill: int = 0
    objective_progress: float = 0.0
    outcome_version: int = 0


@dataclass
class ActionOutcome:
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    successful_block: int = 0
    perfect_dodge: int = 0
    effective_heal: float = 0.0
    kill: int = 0
    objective_progress: float = 0.0

    @property
    def reward(self) -> float:
        value = (
            self.damage_dealt * 0.025
            - self.damage_taken * 0.035
            + self.successful_block * 0.35
            + self.perfect_dodge * 0.65
            + self.effective_heal * 0.018
            + self.kill * 0.8
            + self.objective_progress * 0.12
        )
        return max(-2.0, min(2.0, value))

    @property
    def has_signal(self) -> bool:
        return any(
            (
                self.damage_dealt,
                self.damage_taken,
                self.successful_block,
                self.perfect_dodge,
                self.effective_heal,
                self.kill,
                self.objective_progress,
            )
        )


class GameplayDataCollector:
    def __init__(self, sample_interval: float, player_id: str = "legacy") -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.player_id = player_id or "legacy"
        self.sample_interval = sample_interval
        self.samples: list[TrainingSample] = []
        self._time_since_sample = 0.0
        self._pending_outcomes: dict[int, ActionOutcome] = {}

    def update(
        self,
        dt: float,
        arena_id: int,
        features: Sequence[float],
        action: PlayerAction,
        force: bool = False,
        arena_time: float = 0.0,
    ) -> bool:
        self._time_since_sample += dt
        if not force and self._time_since_sample < self.sample_interval:
            return False
        self._time_since_sample = 0.0 if force else self._time_since_sample % self.sample_interval
        sample = TrainingSample(
            timestamp=time.time(),
            session_id=self.session_id,
            arena_id=arena_id,
            features=validate_features(features),
            action_id=int(action),
            player_id=self.player_id,
            arena_time=arena_time,
            outcome_version=1,
        )
        self.samples.append(sample)
        pending = self._pending_outcomes.pop(int(action), None)
        if pending is not None:
            self._apply_outcome(sample, pending)
        return True

    @staticmethod
    def _apply_outcome(sample: TrainingSample, outcome: ActionOutcome) -> None:
        sample.damage_dealt += outcome.damage_dealt
        sample.damage_taken += outcome.damage_taken
        sample.successful_block += outcome.successful_block
        sample.perfect_dodge += outcome.perfect_dodge
        sample.effective_heal += outcome.effective_heal
        sample.kill += outcome.kill
        sample.objective_progress += outcome.objective_progress
        sample.reward = max(-2.0, min(2.0, sample.reward + outcome.reward))
        sample.outcome_version = 1

    def record_outcome(self, action: PlayerAction, outcome: ActionOutcome) -> None:
        if not outcome.has_signal:
            return
        for sample in reversed(self.samples):
            if sample.action_id == int(action):
                self._apply_outcome(sample, outcome)
                return
        pending = self._pending_outcomes.setdefault(int(action), ActionOutcome())
        pending.damage_dealt += outcome.damage_dealt
        pending.damage_taken += outcome.damage_taken
        pending.successful_block += outcome.successful_block
        pending.perfect_dodge += outcome.perfect_dodge
        pending.effective_heal += outcome.effective_heal
        pending.kill += outcome.kill
        pending.objective_progress += outcome.objective_progress

    def save(self, directory: Path) -> tuple[Path, Path] | None:
        if not self.samples:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / f"session_{self.session_id}.csv"
        meta_path = directory / f"session_{self.session_id}.json"
        outcome_fields = [
            "reward",
            "damage_dealt",
            "damage_taken",
            "successful_block",
            "perfect_dodge",
            "effective_heal",
            "kill",
            "objective_progress",
            "outcome_version",
        ]
        header = ["timestamp", "session_id", "player_id", "arena_id", "arena_time", *FEATURE_NAMES, "action_id", "action_name", *outcome_fields]
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for sample in self.samples:
                action = PlayerAction(sample.action_id)
                writer.writerow(
                    [
                        sample.timestamp,
                        sample.session_id,
                        sample.player_id,
                        sample.arena_id,
                        sample.arena_time,
                        *sample.features,
                        sample.action_id,
                        action.name,
                        sample.reward,
                        sample.damage_dealt,
                        sample.damage_taken,
                        sample.successful_block,
                        sample.perfect_dodge,
                        sample.effective_heal,
                        sample.kill,
                        sample.objective_progress,
                        sample.outcome_version,
                    ]
                )
        counts = {action.name: 0 for action in PlayerAction}
        for sample in self.samples:
            counts[PlayerAction(sample.action_id).name] += 1
        metadata = {
            "session_id": self.session_id,
            "player_id": self.player_id,
            "sample_interval_seconds": self.sample_interval,
            "feature_count": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "action_count": len(PlayerAction),
            "action_names": [action.name for action in PlayerAction],
            "sample_count": len(self.samples),
            "class_counts": counts,
            "normalization": "All model inputs are normalized to [-1, 1].",
            "event_sampling": "Attacks, dash and heal are recorded immediately; continuous actions use the interval.",
            "outcome_schema_version": 1,
            "reward_sum": sum(sample.reward for sample in self.samples),
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return csv_path, meta_path


def samples_by_action(samples: Iterable[TrainingSample]) -> dict[PlayerAction, int]:
    result = {action: 0 for action in PlayerAction}
    for sample in samples:
        result[PlayerAction(sample.action_id)] += 1
    return result
