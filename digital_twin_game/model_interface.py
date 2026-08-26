"""Stable boundary shared by the game, fallback predictor and neural model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from .actions import PlayerAction


FEATURE_NAMES: tuple[str, ...] = (
    "player_health",
    "player_energy",
    "opponent_health",
    "distance_to_opponent",
    "opponent_relative_x",
    "opponent_relative_y",
    "player_velocity_x",
    "player_velocity_y",
    "opponent_velocity_x",
    "opponent_velocity_y",
    "ranged_cooldown",
    "melee_cooldown",
    "dash_cooldown",
    "block_active",
    "heal_available",
    "wall_distance_left",
    "wall_distance_right",
    "wall_distance_top",
    "wall_distance_bottom",
    "healthpack_distance",
    "healthpack_relative_x",
    "healthpack_relative_y",
    "opponent_attacking",
    "recent_damage",
    "last_action",
)

ACTION_COUNT = len(PlayerAction)


@dataclass(frozen=True)
class Prediction:
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.probabilities) != ACTION_COUNT:
            raise ValueError(f"Expected {ACTION_COUNT} probabilities")
        if any(value < 0 for value in self.probabilities):
            raise ValueError("Probabilities cannot be negative")
        if abs(sum(self.probabilities) - 1.0) > 1e-4:
            raise ValueError("Probabilities must sum to 1")

    @property
    def action(self) -> PlayerAction:
        return PlayerAction(max(range(ACTION_COUNT), key=self.probabilities.__getitem__))


class ActionPredictionModel(ABC):
    """Common prediction interface for any 25-input, 10-action model."""

    @abstractmethod
    def predict(self, features: Sequence[float]) -> Prediction:
        """Return probabilities for the 10 actions from 25 normalized features."""

    def reset_history(self) -> None:
        """Reset temporal context; stateless models intentionally do nothing."""


def validate_features(features: Sequence[float]) -> tuple[float, ...]:
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(features)}")
    values = tuple(float(value) for value in features)
    if any(value < -1.0001 or value > 1.0001 for value in values):
        raise ValueError("Every normalized feature must be in the [-1, 1] range")
    return values


class PlaceholderPredictor(ActionPredictionModel):
    """Transparent fallback for UI/integration tests; this is not a neural network."""

    def __init__(self, action_weights: Sequence[float] | None = None) -> None:
        weights = list(action_weights or [1.0] * ACTION_COUNT)
        if len(weights) != ACTION_COUNT or sum(weights) <= 0:
            raise ValueError("Placeholder requires 10 non-negative weights")
        total = sum(max(0.0, value) for value in weights)
        self._probabilities = tuple(max(0.0, value) / total for value in weights)

    def predict(self, features: Sequence[float]) -> Prediction:
        validate_features(features)
        return Prediction(self._probabilities)
