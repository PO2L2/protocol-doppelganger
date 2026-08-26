"""Session-safe loading and temporal preparation of gameplay data."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from .actions import PlayerAction
from .data import TrainingSample
from .model_interface import ACTION_COUNT, FEATURE_NAMES, validate_features


SEQUENCE_LENGTH = 10


@dataclass(frozen=True)
class SessionDataset:
    session_id: str
    player_id: str
    samples: tuple[TrainingSample, ...]


@dataclass(frozen=True)
class SequenceExample:
    states: tuple[tuple[float, ...], ...]
    action_id: int
    session_id: str
    reward: float = 0.0
    player_id: str = "legacy"
    arena_id: int = 0
    arena_time: float = 0.0


@dataclass(frozen=True)
class SessionSplit:
    train: tuple[SessionDataset, ...]
    validation: tuple[SessionDataset, ...]
    test: tuple[SessionDataset, ...]


def load_session_datasets(directory: Path) -> list[SessionDataset]:
    """Load every valid CSV and keep session boundaries intact."""
    grouped: dict[str, list[TrainingSample]] = {}
    for path in sorted(directory.glob("session_*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    session_id = row.get("session_id", "").strip() or path.stem.removeprefix("session_")
                    player_id = row.get("player_id", "").strip() or f"legacy_{session_id}"
                    features = validate_features([float(row[name]) for name in FEATURE_NAMES])
                    action_id = int(row["action_id"])
                    if not 0 <= action_id < ACTION_COUNT:
                        continue
                    grouped.setdefault(session_id, []).append(
                        TrainingSample(
                            timestamp=float(row.get("timestamp", 0.0)),
                            session_id=session_id,
                            arena_id=int(row.get("arena_id", 0)),
                            features=features,
                            action_id=action_id,
                            player_id=player_id,
                            arena_time=float(row.get("arena_time", 0.0) or 0.0),
                            reward=float(row.get("reward", 0.0) or 0.0),
                            damage_dealt=float(row.get("damage_dealt", 0.0) or 0.0),
                            damage_taken=float(row.get("damage_taken", 0.0) or 0.0),
                            successful_block=int(float(row.get("successful_block", 0) or 0)),
                            perfect_dodge=int(float(row.get("perfect_dodge", 0) or 0)),
                            effective_heal=float(row.get("effective_heal", 0.0) or 0.0),
                            kill=int(float(row.get("kill", 0) or 0)),
                            objective_progress=float(row.get("objective_progress", 0.0) or 0.0),
                            outcome_version=int(float(row.get("outcome_version", 0) or 0)),
                        )
                    )
        except (OSError, KeyError, TypeError, ValueError):
            # A damaged old recording must not prevent a new game from starting.
            continue
    return [
        SessionDataset(
            session_id,
            samples[0].player_id,
            _repair_last_action(sorted(samples, key=lambda sample: sample.timestamp)),
        )
        for session_id, samples in sorted(grouped.items())
        if samples
    ]


def _repair_last_action(samples: Sequence[TrainingSample]) -> tuple[TrainingSample, ...]:
    """Old CSVs stored the current label in `last_action`; shift it to prevent leakage."""
    repaired: list[TrainingSample] = []
    previous_action: int | None = None
    for sample in samples:
        features = list(sample.features)
        features[24] = -1.0 if previous_action is None else (previous_action / 9.0) * 2.0 - 1.0
        repaired.append(replace(sample, features=tuple(features)))
        previous_action = sample.action_id
    return tuple(repaired)


def split_by_session(sessions: Sequence[SessionDataset], seed: int = 2026) -> SessionSplit:
    """Deterministically split whole sessions, never adjacent frames."""
    shuffled = list(sessions)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count < 3:
        return SessionSplit(tuple(shuffled), (), ())
    test_count = max(1, round(count * 0.15))
    validation_count = max(1, round(count * 0.15))
    while test_count + validation_count >= count:
        if validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break
    test = tuple(shuffled[:test_count])
    validation = tuple(shuffled[test_count : test_count + validation_count])
    train = tuple(shuffled[test_count + validation_count :])
    return SessionSplit(train, validation, test)


def split_by_player(sessions: Sequence[SessionDataset], seed: int = 2026) -> SessionSplit:
    """Put every session of the same player into exactly one split."""
    grouped: dict[str, list[SessionDataset]] = {}
    for session in sessions:
        grouped.setdefault(session.player_id, []).append(session)
    player_ids = list(grouped)
    if len(player_ids) < 3:
        return split_by_session(sessions, seed)
    random.Random(seed).shuffle(player_ids)
    test_count = max(1, round(len(player_ids) * 0.15))
    validation_count = max(1, round(len(player_ids) * 0.15))
    while test_count + validation_count >= len(player_ids):
        validation_count = max(1, validation_count - 1)
        if test_count + validation_count < len(player_ids):
            break
        test_count = max(1, test_count - 1)
    test_players = set(player_ids[:test_count])
    validation_players = set(player_ids[test_count : test_count + validation_count])
    train_players = set(player_ids[test_count + validation_count :])
    return SessionSplit(
        tuple(session for session in sessions if session.player_id in train_players),
        tuple(session for session in sessions if session.player_id in validation_players),
        tuple(session for session in sessions if session.player_id in test_players),
    )


def _context_reward(current: TrainingSample, following: TrainingSample | None) -> float:
    """Small bounded reward for conservative offline reinforcement learning."""
    if current.outcome_version > 0:
        return current.reward
    action = PlayerAction(current.action_id)
    values = current.features
    opponent_damage = 0.0
    player_damage = 0.0
    if following is not None:
        opponent_damage = max(0.0, values[2] - following.features[2])
        player_damage = max(0.0, values[0] - following.features[0])
    reward = opponent_damage * 3.0 - player_damage * 2.4
    opponent_attacking = values[22] > 0.2
    recent_damage = values[23] > 0.15
    if action in (PlayerAction.RANGED_ATTACK, PlayerAction.MELEE_ATTACK) and opponent_damage > 0:
        reward += 0.25
    elif action == PlayerAction.BLOCK and opponent_attacking:
        reward += 0.18
    elif action == PlayerAction.DASH and (opponent_attacking or recent_damage):
        reward += 0.16
    elif action == PlayerAction.HEAL and values[0] < 0.55:
        reward += 0.2
    elif action == PlayerAction.RETREAT and values[0] < 0.3:
        reward += 0.08
    elif action == PlayerAction.APPROACH and values[3] > 0.45:
        reward += 0.06
    elif action == PlayerAction.IDLE:
        reward -= 0.025
    return max(-1.0, min(1.0, reward))


def build_sequences(
    sessions: Iterable[SessionDataset], sequence_length: int = SEQUENCE_LENGTH
) -> list[SequenceExample]:
    """Build padded rolling windows without ever crossing a session boundary."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    examples: list[SequenceExample] = []
    for session in sessions:
        if not session.samples:
            continue
        for index, sample in enumerate(session.samples):
            start = max(0, index - sequence_length + 1)
            states = [row.features for row in session.samples[start : index + 1]]
            states = [states[0]] * (sequence_length - len(states)) + states
            following = session.samples[index + 1] if index + 1 < len(session.samples) else None
            examples.append(
                SequenceExample(
                    states=tuple(states),
                    action_id=sample.action_id,
                    session_id=session.session_id,
                    reward=_context_reward(sample, following),
                    player_id=session.player_id,
                    arena_id=sample.arena_id,
                    arena_time=sample.arena_time,
                )
            )
    return examples


def session_from_samples(samples: Iterable[TrainingSample]) -> SessionDataset | None:
    rows = sorted(list(samples), key=lambda sample: sample.timestamp)
    if not rows:
        return None
    return SessionDataset(rows[0].session_id, rows[0].player_id, _repair_last_action(rows))


def class_counts(examples: Iterable[SequenceExample]) -> tuple[int, ...]:
    counts = [0] * ACTION_COUNT
    for example in examples:
        counts[example.action_id] += 1
    return tuple(counts)


def balanced_sample_weights(examples: Sequence[SequenceExample]) -> list[float]:
    """Moderate square-root balancing without erasing the real play distribution."""
    counts = class_counts(examples)
    non_empty = [count for count in counts if count]
    if not non_empty:
        return []
    maximum = max(non_empty)
    per_class = [min(4.0, math.sqrt(maximum / count)) if count else 0.0 for count in counts]
    return [per_class[example.action_id] for example in examples]


def save_split_manifest(path: Path, split: SessionSplit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": "player_level_70_15_15",
        "sequence_length": SEQUENCE_LENGTH,
        "train_sessions": [session.session_id for session in split.train],
        "validation_sessions": [session.session_id for session in split.validation],
        "test_sessions": [session.session_id for session in split.test],
        "train_players": sorted({session.player_id for session in split.train}),
        "validation_players": sorted({session.player_id for session in split.validation}),
        "test_players": sorted({session.player_id for session in split.test}),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
