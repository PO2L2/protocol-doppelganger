from pathlib import Path


TITLE = "Протокол: Двойник"
WIDTH = 1280
HEIGHT = 720
FPS = 120

ARENA_MARGIN = 48
HUD_HEIGHT = 76
PLAY_RECT = (ARENA_MARGIN, HUD_HEIGHT + 20, WIDTH - ARENA_MARGIN * 2, HEIGHT - HUD_HEIGHT - 68)

PLAYER_SPEED = 270.0
PLAYER_RADIUS = 18
PLAYER_MAX_HEALTH = 100.0
PLAYER_MAX_ENERGY = 100.0

SAMPLE_INTERVAL = 0.2  # five samples per second; avoids near-duplicate frame data
TRAINING_ROUND_SECONDS = 35.0
TWIN_INTRO_SECONDS = 2.8
SYNC_WINDOW_SAMPLES = 30

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

COLORS = {
    "background": (8, 12, 24),
    "panel": (17, 25, 43),
    "grid": (25, 38, 62),
    "text": (225, 235, 247),
    "muted": (137, 158, 184),
    "player": (54, 224, 189),
    "player_glow": (22, 107, 100),
    "enemy": (255, 91, 115),
    "twin": (177, 90, 255),
    "accent": (74, 163, 255),
    "warning": (255, 196, 77),
    "health": (68, 214, 122),
    "energy": (69, 143, 255),
    "white": (255, 255, 255),
}
