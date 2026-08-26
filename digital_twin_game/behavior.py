from __future__ import annotations

from dataclasses import dataclass

from .actions import PlayerAction
from .data import TrainingSample, samples_by_action


@dataclass
class BehaviorProfile:
    style_name: str
    aggression: float
    mobility: float
    defense: float
    preferred_range: float
    predictability: float
    favorite_action: PlayerAction
    action_rates: dict[PlayerAction, float]
    sample_count: int

    @classmethod
    def from_samples(cls, samples: list[TrainingSample]) -> "BehaviorProfile":
        if not samples:
            uniform = {action: 0.0 for action in PlayerAction}
            return cls("Недостаточно данных", 0, 0, 0, 0.5, 0, PlayerAction.IDLE, uniform, 0)

        counts = samples_by_action(samples)
        total = len(samples)
        rates = {action: count / total for action, count in counts.items()}
        aggression = min(1.0, rates[PlayerAction.APPROACH] + rates[PlayerAction.RANGED_ATTACK] + rates[PlayerAction.MELEE_ATTACK])
        mobility = min(1.0, rates[PlayerAction.STRAFE_LEFT] + rates[PlayerAction.STRAFE_RIGHT] + rates[PlayerAction.DASH])
        defense = min(1.0, rates[PlayerAction.RETREAT] + rates[PlayerAction.BLOCK] + rates[PlayerAction.HEAL])
        distances = [sample.features[3] for sample in samples if sample.action_id in (PlayerAction.RANGED_ATTACK, PlayerAction.MELEE_ATTACK)]
        preferred_range = sum(distances) / len(distances) if distances else 0.5
        favorite = max(counts, key=counts.get)

        # A transparent baseline estimate, not a claim about neural-network accuracy.
        predictability = max(rates.values())
        if aggression >= defense + 0.12:
            style = "Агрессивный охотник"
        elif defense >= aggression + 0.12:
            style = "Осторожный тактик"
        elif mobility > 0.42:
            style = "Подвижный дуэлянт"
        elif rates[PlayerAction.RANGED_ATTACK] > rates[PlayerAction.MELEE_ATTACK] * 1.5:
            style = "Дальний стрелок"
        else:
            style = "Сбалансированный боец"
        return cls(style, aggression, mobility, defense, preferred_range, predictability, favorite, rates, total)

    def placeholder_weights(self) -> list[float]:
        """Class priors for the non-neural temporary twin controller."""
        return [max(0.02, self.action_rates[action]) for action in PlayerAction]

