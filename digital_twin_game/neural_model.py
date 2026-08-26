from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

from .model_interface import ACTION_COUNT, FEATURE_NAMES, ActionPredictionModel, Prediction, validate_features


@dataclass(frozen=True)
class TrainingMetrics:
    samples: int
    epochs: int
    loss: float
    accuracy: float


class NeuralActionModel(ActionPredictionModel):
    """A real 25-64-32-10 MLP with ReLU layers and softmax output."""

    LAYERS = (len(FEATURE_NAMES), 64, 32, ACTION_COUNT)

    def __init__(self, seed: int = 2026) -> None:
        generator = random.Random(seed)
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []
        for input_size, output_size in zip(self.LAYERS, self.LAYERS[1:]):
            scale = math.sqrt(2.0 / input_size)
            self.weights.append(
                [[generator.gauss(0.0, scale) for _ in range(input_size)] for _ in range(output_size)]
            )
            self.biases.append([0.0] * output_size)

    @property
    def parameter_count(self) -> int:
        return sum(len(row) for matrix in self.weights for row in matrix) + sum(len(values) for values in self.biases)

    @staticmethod
    def _dense(inputs: Sequence[float], weights: list[list[float]], biases: list[float]) -> list[float]:
        return [sum(weight * value for weight, value in zip(row, inputs)) + bias for row, bias in zip(weights, biases)]

    @staticmethod
    def _softmax(logits: Sequence[float]) -> list[float]:
        maximum = max(logits)
        exponents = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in logits]
        total = sum(exponents) or 1.0
        return [value / total for value in exponents]

    def _forward(self, features: Sequence[float]) -> tuple[list[list[float]], list[list[float]]]:
        activations: list[list[float]] = [list(features)]
        preactivations: list[list[float]] = []
        current = list(features)
        for layer_index in range(2):
            raw = self._dense(current, self.weights[layer_index], self.biases[layer_index])
            preactivations.append(raw)
            current = [max(0.0, value) for value in raw]
            activations.append(current)
        logits = self._dense(current, self.weights[2], self.biases[2])
        preactivations.append(logits)
        activations.append(self._softmax(logits))
        return activations, preactivations

    def predict(self, features: Sequence[float]) -> Prediction:
        values = validate_features(features)
        probabilities = self._forward(values)[0][-1]
        return Prediction(tuple(probabilities))

    def fit(
        self,
        samples: Iterable,
        epochs: int = 10,
        learning_rate: float = 0.012,
        maximum_samples: int = 800,
    ) -> TrainingMetrics:
        rows = list(samples)[-maximum_samples:]
        if not rows:
            return TrainingMetrics(0, 0, 0.0, 0.0)
        generator = random.Random(991)
        for _ in range(epochs):
            generator.shuffle(rows)
            for sample in rows:
                features = validate_features(sample.features)
                target = int(sample.action_id)
                activations, preactivations = self._forward(features)
                deltas: list[list[float]] = [[], [], list(activations[-1])]
                deltas[2][target] -= 1.0
                for layer_index in (2, 1):
                    previous_delta = [
                        sum(self.weights[layer_index][output][input_index] * deltas[layer_index][output] for output in range(len(deltas[layer_index])))
                        for input_index in range(len(activations[layer_index]))
                    ]
                    deltas[layer_index - 1] = [
                        value if preactivations[layer_index - 1][index] > 0 else 0.0
                        for index, value in enumerate(previous_delta)
                    ]
                for layer_index in range(3):
                    layer_input = activations[layer_index]
                    for output_index, delta in enumerate(deltas[layer_index]):
                        row = self.weights[layer_index][output_index]
                        for input_index, input_value in enumerate(layer_input):
                            row[input_index] -= learning_rate * delta * input_value
                        self.biases[layer_index][output_index] -= learning_rate * delta

        total_loss = 0.0
        correct = 0
        for sample in rows:
            probabilities = self.predict(sample.features).probabilities
            target = int(sample.action_id)
            total_loss -= math.log(max(1e-9, probabilities[target]))
            correct += int(max(range(ACTION_COUNT), key=probabilities.__getitem__) == target)
        return TrainingMetrics(len(rows), epochs, total_loss / len(rows), correct / len(rows))

    def save(self, path: Path, metrics: TrainingMetrics | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "architecture": self.LAYERS,
            "parameter_count": self.parameter_count,
            "weights": self.weights,
            "biases": self.biases,
            "metrics": metrics.__dict__ if metrics else None,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NeuralActionModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if tuple(payload.get("architecture", ())) != cls.LAYERS:
            raise ValueError("Unsupported neural-network architecture")
        model = cls()
        model.weights = payload["weights"]
        model.biases = payload["biases"]
        return model

