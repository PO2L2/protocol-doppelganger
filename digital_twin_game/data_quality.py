"""Fast, transparent quality checks for a gameplay calibration dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .actions import ACTION_LABELS_RU, PlayerAction
from .data import TrainingSample


@dataclass(frozen=True)
class DataQualityReport:
    sample_count: int
    class_counts: tuple[int, ...]
    covered_classes: int
    insufficient_actions: tuple[PlayerAction, ...]
    missing_actions: tuple[PlayerAction, ...]
    duplicate_ratio: float
    real_reward_ratio: float
    score: float

    @property
    def status(self) -> str:
        if self.score >= 0.8:
            return "ОТЛИЧНО"
        if self.score >= 0.58:
            return "ХОРОШО"
        if self.score >= 0.35:
            return "НУЖНЫ ДАННЫЕ"
        return "МАЛО ДАННЫХ"

    @property
    def warning(self) -> str:
        actions = self.missing_actions or self.insufficient_actions
        if not actions:
            return "Все десять действий представлены достаточно"
        names = ", ".join(ACTION_LABELS_RU[action].lower() for action in actions[:4])
        suffix = "…" if len(actions) > 4 else ""
        return f"Нужно собрать: {names}{suffix}"


def analyze_data_quality(samples: Iterable[TrainingSample], minimum_per_action: int = 8) -> DataQualityReport:
    rows = list(samples)
    counts = [0] * len(PlayerAction)
    unique: set[tuple] = set()
    real_rewards = 0
    for sample in rows:
        counts[int(sample.action_id)] += 1
        unique.add((sample.action_id, *(round(value, 3) for value in sample.features)))
        real_rewards += int(getattr(sample, "outcome_version", 0) > 0)
    missing = tuple(PlayerAction(index) for index, count in enumerate(counts) if count == 0)
    insufficient = tuple(PlayerAction(index) for index, count in enumerate(counts) if count < minimum_per_action)
    duplicate_ratio = 1.0 - len(unique) / max(1, len(rows))
    reward_ratio = real_rewards / max(1, len(rows))
    coverage = sum(count >= minimum_per_action for count in counts) / len(counts)
    volume = min(1.0, len(rows) / 350)
    diversity = max(0.0, 1.0 - duplicate_ratio)
    score = max(0.0, min(1.0, coverage * 0.55 + volume * 0.2 + diversity * 0.15 + reward_ratio * 0.1))
    return DataQualityReport(
        len(rows),
        tuple(counts),
        sum(count > 0 for count in counts),
        insufficient,
        missing,
        duplicate_ratio,
        reward_ratio,
        score,
    )
