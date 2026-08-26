"""Retrain the general and personal digital-twin models from saved sessions."""

from pathlib import Path

from digital_twin_game.config import DATA_DIR, MODEL_DIR
from digital_twin_game.sequence_data import load_session_datasets
from digital_twin_game.torch_model import train_gru_pipeline


def main() -> None:
    sessions = load_session_datasets(DATA_DIR)
    if not sessions:
        raise SystemExit("Нет сохранённых сессий в папке data.")
    current = max(sessions, key=lambda session: session.samples[-1].timestamp)
    _model, metrics = train_gru_pipeline(DATA_DIR, current.samples, MODEL_DIR)
    print(f"Сессий: {len(sessions)}; текущая: {current.session_id}")
    print(f"Устройство: {metrics.device}")
    print(f"Персональная точность: {metrics.accuracy * 100:.1f}%")
    print(f"Тестовая точность: {metrics.test_accuracy * 100:.1f}%")
    print(f"Macro-F1: {metrics.macro_f1 * 100:.1f}%")
    print(f"Offline RL: {'принято' if metrics.rl_applied else 'отклонено безопасной проверкой'}")
    print(f"Модель: {Path(MODEL_DIR / 'personal_gru.pt').resolve()}")


if __name__ == "__main__":
    main()
