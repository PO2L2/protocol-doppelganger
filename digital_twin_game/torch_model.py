"""PyTorch GRU model, evaluation and two-stage personalization pipeline."""

from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError:  # The pure-Python MLP remains an emergency fallback.
    torch = None
    F = None
    nn = None
    DataLoader = Dataset = WeightedRandomSampler = object

# PyTorch 2.13 + cuDNN RNN currently terminates some Windows processes with
# 0xC0000409 after GRU backpropagation. Native CUDA GRU kernels are stable and
# easily fast enough for this small model, so keep cuDNN off for this module.
if torch is not None:
    torch.backends.cudnn.enabled = False

from .model_interface import ACTION_COUNT, FEATURE_NAMES, ActionPredictionModel, Prediction, validate_features
from .data_quality import analyze_data_quality
from .sequence_data import (
    SEQUENCE_LENGTH,
    SequenceExample,
    balanced_sample_weights,
    build_sequences,
    class_counts,
    load_session_datasets,
    save_split_manifest,
    session_from_samples,
    split_by_player,
)


def _empty_matrix() -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in range(ACTION_COUNT)) for _ in range(ACTION_COUNT))


@dataclass(frozen=True)
class TorchTrainingMetrics:
    samples: int
    epochs: int
    loss: float
    accuracy: float
    test_accuracy: float = 0.0
    macro_f1: float = 0.0
    confusion_matrix: tuple[tuple[int, ...], ...] = field(default_factory=_empty_matrix)
    sequence_length: int = SEQUENCE_LENGTH
    parameter_count: int = 0
    device: str = "CPU"
    train_sessions: int = 0
    validation_sessions: int = 0
    test_sessions: int = 0
    general_test_accuracy: float = 0.0
    general_player_accuracy: float = 0.0
    class_counts: tuple[int, ...] = field(default_factory=lambda: (0,) * ACTION_COUNT)
    rl_applied: bool = False
    rl_reward_before: float = 0.0
    rl_reward_after: float = 0.0
    train_players: int = 0
    validation_players: int = 0
    test_players: int = 0
    quality_score: float = 0.0
    insufficient_actions: tuple[int, ...] = ()
    training_history: tuple[dict, ...] = ()
    error_examples: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Evaluation:
    loss: float
    accuracy: float
    macro_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]
    error_examples: tuple[dict, ...] = ()


if nn is not None:

    class BehaviorGRUNetwork(nn.Module):
        """25 inputs over ten moments -> two GRU layers -> 10 actions."""

        def __init__(self) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(len(FEATURE_NAMES))
            self.gru = nn.GRU(
                input_size=len(FEATURE_NAMES),
                hidden_size=64,
                num_layers=2,
                dropout=0.15,
                batch_first=True,
            )
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, ACTION_COUNT))

        def forward(self, states):
            normalized = self.normalization(states)
            sequence, _ = self.gru(normalized)
            return self.head(sequence[:, -1, :])


    class _SequenceDataset(Dataset):
        def __init__(self, examples: Sequence[SequenceExample]) -> None:
            self.examples = list(examples)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int):
            example = self.examples[index]
            return (
                torch.tensor(example.states, dtype=torch.float32),
                torch.tensor(example.action_id, dtype=torch.long),
                torch.tensor(example.reward, dtype=torch.float32),
            )


else:

    class BehaviorGRUNetwork:  # pragma: no cover - reached only without an installed dependency
        pass


def torch_available() -> bool:
    return torch is not None


def _best_device():
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchSequenceModel(ActionPredictionModel):
    """Stateful inference wrapper that remembers the latest ten game states."""

    ARCHITECTURE = "GRU 25→64×2→32→10 / temporal-v3"

    def __init__(self, network=None, device=None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        self.device = device or _best_device()
        self.network = (network or BehaviorGRUNetwork()).to(self.device)
        self.network.eval()
        self._history: deque[tuple[float, ...]] = deque(maxlen=SEQUENCE_LENGTH)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.network.parameters())

    def reset_history(self) -> None:
        self._history.clear()

    def predict(self, features: Sequence[float]) -> Prediction:
        values = validate_features(features)
        self._history.append(values)
        history = list(self._history)
        history = [history[0]] * (SEQUENCE_LENGTH - len(history)) + history
        states = torch.tensor([history], dtype=torch.float32, device=self.device)
        self.network.eval()
        with torch.inference_mode():
            probabilities = torch.softmax(self.network(states), dim=1)[0].detach().cpu().tolist()
        return Prediction(tuple(float(value) for value in probabilities))

    def save(self, path: Path, metrics: TorchTrainingMetrics | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {name: value.detach().cpu() for name, value in self.network.state_dict().items()}
        torch.save(
            {
                "architecture": self.ARCHITECTURE,
                "sequence_length": SEQUENCE_LENGTH,
                "feature_count": len(FEATURE_NAMES),
                "action_count": ACTION_COUNT,
                "state_dict": state,
                "metrics": asdict(metrics) if metrics else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device=None) -> "TorchSequenceModel":
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        target = device or _best_device()
        payload = torch.load(path, map_location=target, weights_only=True)
        if payload.get("architecture") != cls.ARCHITECTURE:
            raise ValueError("Unsupported GRU checkpoint")
        network = BehaviorGRUNetwork()
        network.load_state_dict(payload["state_dict"])
        return cls(network, target)


def _data_loader(
    examples: Sequence[SequenceExample], batch_size: int, balanced: bool, shuffle: bool = False
) -> DataLoader:
    dataset = _SequenceDataset(examples)
    sampler = None
    should_shuffle = shuffle and not balanced
    if balanced and examples:
        weights = balanced_sample_weights(examples)
        generator = torch.Generator().manual_seed(2026)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        should_shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=should_shuffle, sampler=sampler, num_workers=0)


def _train_supervised(
    model: TorchSequenceModel,
    train_examples: Sequence[SequenceExample],
    validation_examples: Sequence[SequenceExample],
    epochs: int,
    learning_rate: float,
    stage: str,
    progress_callback=None,
) -> tuple[int, float, list[dict]]:
    if not train_examples:
        return 0, 0.0, []
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = _data_loader(train_examples, batch_size=128, balanced=True)
    best_state = copy.deepcopy(model.network.state_dict())
    best_loss = math.inf
    stale_epochs = 0
    completed = 0
    last_loss = 0.0
    history: list[dict] = []
    for epoch_index in range(epochs):
        completed += 1
        model.network.train()
        running_loss = 0.0
        seen = 0
        for states, actions, _rewards in loader:
            states = states.to(model.device, non_blocking=True)
            actions = actions.to(model.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model.network(states)
            loss = F.cross_entropy(logits, actions, label_smoothing=0.025)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(actions)
            seen += len(actions)
        last_loss = running_loss / max(1, seen)
        evaluation = evaluate_model(model, validation_examples or train_examples)
        monitored = evaluation.loss
        point = {
            "stage": stage,
            "epoch": epoch_index + 1,
            "loss": evaluation.loss,
            "accuracy": evaluation.accuracy,
            "macro_f1": evaluation.macro_f1,
        }
        history.append(point)
        if progress_callback is not None:
            progress_callback(stage, epoch_index + 1, epochs, point)
        if monitored < best_loss - 1e-4:
            best_loss = monitored
            best_state = copy.deepcopy(model.network.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 3:
                break
    model.network.load_state_dict(best_state)
    model.network.eval()
    return completed, last_loss, history


def evaluate_model(model: TorchSequenceModel, examples: Sequence[SequenceExample]) -> Evaluation:
    if not examples:
        return Evaluation(0.0, 0.0, 0.0, _empty_matrix(), ())
    matrix = [[0 for _ in range(ACTION_COUNT)] for _ in range(ACTION_COUNT)]
    total_loss = 0.0
    total = 0
    cursor = 0
    errors: list[dict] = []
    model.network.eval()
    with torch.inference_mode():
        for states, actions, _rewards in _data_loader(examples, batch_size=256, balanced=False):
            states = states.to(model.device, non_blocking=True)
            actions = actions.to(model.device, non_blocking=True)
            logits = model.network(states)
            total_loss += float(F.cross_entropy(logits, actions, reduction="sum").detach())
            predictions = logits.argmax(dim=1).detach().cpu().tolist()
            actual = actions.detach().cpu().tolist()
            for offset, (target, predicted) in enumerate(zip(actual, predictions)):
                matrix[target][predicted] += 1
                if predicted != target and len(errors) < 240:
                    example = examples[cursor + offset]
                    errors.append(
                        {
                            "session_id": example.session_id,
                            "player_id": example.player_id,
                            "arena_id": example.arena_id,
                            "arena_time": example.arena_time,
                            "actual": target,
                            "predicted": predicted,
                        }
                    )
            total += len(actual)
            cursor += len(actual)
    correct = sum(matrix[index][index] for index in range(ACTION_COUNT))
    f1_scores: list[float] = []
    for action in range(ACTION_COUNT):
        support = sum(matrix[action])
        if not support:
            continue
        true_positive = matrix[action][action]
        false_positive = sum(matrix[row][action] for row in range(ACTION_COUNT) if row != action)
        false_negative = support - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return Evaluation(
        loss=total_loss / max(1, total),
        accuracy=correct / max(1, total),
        macro_f1=sum(f1_scores) / max(1, len(f1_scores)),
        confusion_matrix=tuple(tuple(row) for row in matrix),
        error_examples=tuple(errors),
    )


def _offline_objective(model: TorchSequenceModel, examples: Sequence[SequenceExample]) -> float:
    if not examples:
        return 0.0
    total = 0.0
    count = 0
    model.network.eval()
    with torch.inference_mode():
        for states, actions, rewards in _data_loader(examples, 256, balanced=False):
            states = states.to(model.device)
            probabilities = torch.softmax(model.network(states), dim=1).detach().cpu()
            chosen = probabilities.gather(1, actions[:, None]).squeeze(1)
            total += float((chosen * rewards).sum())
            count += len(actions)
    return total / max(1, count)


def _offline_reinforcement_finetune(
    model: TorchSequenceModel,
    examples: Sequence[SequenceExample],
    validation_examples: Sequence[SequenceExample],
    epochs: int = 2,
) -> tuple[bool, float, float, int]:
    """Conservative logged-data policy gradient with KL and imitation anchors."""
    if len(examples) < 32:
        return False, 0.0, 0.0, 0
    reward_values = [example.reward for example in examples]
    reward_mean = sum(reward_values) / len(reward_values)
    reward_variance = sum((value - reward_mean) ** 2 for value in reward_values) / len(reward_values)
    if reward_variance < 1e-6:
        return False, 0.0, 0.0, 0
    original_state = copy.deepcopy(model.network.state_dict())
    reference = copy.deepcopy(model.network).to(model.device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    baseline = evaluate_model(model, validation_examples or examples)
    before = _offline_objective(model, examples)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=1.5e-4, weight_decay=1e-4)
    loader = _data_loader(examples, 128, balanced=False, shuffle=True)
    reward_std = math.sqrt(reward_variance) + 1e-6
    for _ in range(epochs):
        model.network.train()
        for states, actions, rewards in loader:
            states = states.to(model.device)
            actions = actions.to(model.device)
            advantages = ((rewards - reward_mean) / reward_std).to(model.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.network(states)
            log_probabilities = F.log_softmax(logits, dim=1)
            selected = log_probabilities.gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                reference_probabilities = torch.softmax(reference(states), dim=1)
            policy_loss = -(advantages * selected).mean()
            anchor_loss = F.cross_entropy(logits, actions)
            kl_loss = F.kl_div(log_probabilities, reference_probabilities, reduction="batchmean")
            loss = policy_loss + 0.15 * anchor_loss + 0.08 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 0.7)
            optimizer.step()
    model.network.eval()
    after = _offline_objective(model, examples)
    updated = evaluate_model(model, validation_examples or examples)
    accepted = after >= before and updated.accuracy >= baseline.accuracy - 0.03
    if not accepted:
        model.network.load_state_dict(original_state)
        after = before
    return accepted, before, after, epochs


def train_gru_pipeline(
    data_directory: Path,
    current_samples: Iterable,
    model_directory: Path,
    demo_mode: bool = False,
    progress_callback=None,
) -> tuple[TorchSequenceModel, TorchTrainingMetrics]:
    """Pretrain on prior sessions, personalize, evaluate, then safely try offline RL."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)

    rows = list(current_samples)
    current_session = session_from_samples(rows)
    prior_sessions = load_session_datasets(data_directory)
    if current_session is not None:
        prior_sessions = [session for session in prior_sessions if session.session_id != current_session.session_id]
    current_player_id = current_session.player_id if current_session else "legacy"
    other_players = [session for session in prior_sessions if session.player_id != current_player_id]
    general_pool = other_players or prior_sessions
    split = split_by_player(general_pool)
    save_split_manifest(model_directory / "session_split.json", split)

    train_examples = build_sequences(split.train)
    validation_examples = build_sequences(split.validation)
    test_examples = build_sequences(split.test)
    model = TorchSequenceModel()
    general_path = model_directory / "general_gru.pt"
    if general_path.exists():
        try:
            model = TorchSequenceModel.load(general_path, model.device)
        except (OSError, RuntimeError, ValueError, KeyError):
            model = TorchSequenceModel(device=model.device)

    general_epochs = 0
    training_history: list[dict] = []
    if train_examples:
        general_epochs, _, general_history = _train_supervised(
            model,
            train_examples,
            validation_examples,
            epochs=4 if demo_mode else 18,
            learning_rate=8e-4,
            stage="general",
            progress_callback=progress_callback,
        )
        training_history.extend(general_history)
        model.save(general_path)
    general_test = evaluate_model(model, test_examples or validation_examples)

    previous_personal_sessions = [session for session in prior_sessions if session.player_id == current_player_id]
    current_examples = build_sequences([current_session]) if current_session else []
    older_personal_examples = build_sequences(previous_personal_sessions)
    personal_examples = older_personal_examples + current_examples
    if current_examples:
        boundary = max(1, int(len(current_examples) * 0.8))
        personal_train = older_personal_examples + current_examples[:boundary]
        personal_validation = current_examples[boundary:] or current_examples[-min(32, len(current_examples)) :]
    else:
        personal_train = []
        personal_validation = []
    general_player_evaluation = evaluate_model(model, personal_validation or personal_train)
    personal_epochs, _, personal_history = _train_supervised(
        model,
        personal_train,
        personal_validation,
        epochs=2 if demo_mode else 6,
        learning_rate=3e-4,
        stage="personal",
        progress_callback=progress_callback,
    )
    training_history.extend(personal_history)
    personal_evaluation = evaluate_model(model, personal_validation or personal_train)

    rl_applied, reward_before, reward_after, rl_epochs = _offline_reinforcement_finetune(
        model,
        personal_train,
        personal_validation,
        epochs=1 if demo_mode else 2,
    )
    personal_evaluation = evaluate_model(model, personal_validation or personal_train)
    final_test = evaluate_model(model, test_examples or personal_validation or personal_train)
    device_name = torch.cuda.get_device_name(0) if model.device.type == "cuda" else "CPU"
    quality = analyze_data_quality(rows)
    metrics = TorchTrainingMetrics(
        samples=len(personal_examples),
        epochs=general_epochs + personal_epochs + rl_epochs,
        loss=final_test.loss,
        accuracy=personal_evaluation.accuracy,
        test_accuracy=final_test.accuracy,
        macro_f1=final_test.macro_f1,
        confusion_matrix=final_test.confusion_matrix,
        parameter_count=model.parameter_count,
        device=device_name,
        train_sessions=len(split.train),
        validation_sessions=len(split.validation),
        test_sessions=len(split.test),
        general_test_accuracy=general_test.accuracy,
        general_player_accuracy=general_player_evaluation.accuracy,
        class_counts=class_counts(personal_examples),
        rl_applied=rl_applied,
        rl_reward_before=reward_before,
        rl_reward_after=reward_after,
        train_players=len({session.player_id for session in split.train}),
        validation_players=len({session.player_id for session in split.validation}),
        test_players=len({session.player_id for session in split.test}),
        quality_score=quality.score,
        insufficient_actions=tuple(int(action) for action in quality.insufficient_actions),
        training_history=tuple(training_history),
        error_examples=final_test.error_examples,
    )
    model.save(model_directory / "personal_gru.pt", metrics)
    (model_directory / "ai_metrics.json").write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history_path = model_directory / "metrics_history.json"
    history_payload: list[dict] = []
    if history_path.exists():
        try:
            history_payload = list(json.loads(history_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            history_payload = []
    record = {
        "timestamp": time.time(),
        "player_id": current_player_id,
        "session_id": current_session.session_id if current_session else "",
        "general_accuracy": general_player_evaluation.accuracy,
        "personal_accuracy": personal_evaluation.accuracy,
        "test_accuracy": final_test.accuracy,
        "macro_f1": final_test.macro_f1,
        "loss": final_test.loss,
        "quality_score": quality.score,
    }
    history_payload = [item for item in history_payload if item.get("session_id") != record["session_id"]]
    history_payload.append(record)
    history_path.write_text(json.dumps(history_payload[-100:], ensure_ascii=False, indent=2), encoding="utf-8")
    model.reset_history()
    return model, metrics
