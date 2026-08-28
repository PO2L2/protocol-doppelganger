from __future__ import annotations

import math
import random
import json
import threading
import ctypes

import pygame

from .actions import ACTION_LABELS_RU, PlayerAction
from .analysis import ComboTracker, PositionHeatmap
from .abilities import ABILITY_INFO, AbilitySystem, AbilityType
from .arena import Arena
from .background import AnimatedConstellationBackground
from .behavior import BehaviorProfile
from .calibration import CalibrationChallenge
from .config import (
    COLORS,
    DATA_DIR,
    FPS,
    HEIGHT,
    MODEL_DIR,
    SAMPLE_INTERVAL,
    SYNC_WINDOW_SAMPLES,
    TITLE,
    TRAINING_ROUND_SECONDS,
    TWIN_INTRO_SECONDS,
    WIDTH,
)
from .data import ActionOutcome, GameplayDataCollector
from .data_quality import DataQualityReport, analyze_data_quality
from .editor import GAME_MODE_NAMES, ArenaEditor, ArenaLayout
from .entities import Decoy, Enemy, Player, Projectile, SlashEffect, TrapHazard, TurretHazard, segment_intersects_circle
from .features import build_feature_vector, classify_movement_action
from .fx import AudioManager, CombatEffects
from .hidpi import blit as logical_blit, draw as logical_draw, register_surface, scale_rect, unregister_surface
from .lab import SessionSummary, load_sessions
from .model_interface import ActionPredictionModel, PlaceholderPredictor
from .neural_model import NeuralActionModel, TrainingMetrics
from .objectives import TrainingObjective
from .players import PLAYER_NAME_MAX_LENGTH, PlayerProfileRecord, PlayerRegistry
from .replay import ReplayRecorder
from .tournament import TournamentFighter, TournamentMatch
from .ui import draw_bar, draw_metric, draw_panel, draw_text, get_font
from .upgrades import UPGRADE_INFO, UpgradeType
from .viewport import DisplayViewport
from .weapons import WEAPON_SPECS, WeaponType

try:
    from .torch_model import TorchTrainingMetrics, train_gru_pipeline
except ImportError:
    TorchTrainingMetrics = TrainingMetrics
    train_gru_pipeline = None


INTERACTIVE_TUTORIAL_STEPS = [
    ("ДВИЖЕНИЕ", "Удерживайте WASD и перемещайтесь по арене", "WASD"),
    ("ДАЛЬНЯЯ АТАКА", "Попадите по учебной цели три раза", "ЛКМ"),
    ("БЛИЖНЯЯ АТАКА", "Подойдите к цели и нанесите ближний удар", "ПКМ"),
    ("БЛОК", "Удерживайте блок в течение одной секунды", "SHIFT"),
    ("РЫВОК", "Выберите направление движения и выполните рывок", "SPACE"),
    ("СПОСОБНОСТЬ", "Используйте первую способность", "1"),
]


def _enable_windows_dpi_awareness() -> None:
    """Keep Windows display scaling from making fullscreen output blurry."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class DigitalTwinGame:
    def __init__(self, action_model: ActionPredictionModel | None = None) -> None:
        _enable_windows_dpi_awareness()
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.fullscreen = True
        self.windowed_size = (WIDTH, HEIGHT)
        desktop_sizes = pygame.display.get_desktop_sizes()
        self.desktop_size = desktop_sizes[0] if desktop_sizes else (WIDTH, HEIGHT)
        self.display_surface = pygame.display.set_mode(self.desktop_size, pygame.FULLSCREEN | pygame.DOUBLEBUF)
        try:
            pygame.scrap.init()
        except pygame.error:
            pass
        self.viewport = DisplayViewport((WIDTH, HEIGHT), self.display_surface.get_size())
        self._create_render_surfaces()
        self.clock = pygame.time.Clock()
        self.animated_background = AnimatedConstellationBackground()
        self.frame_dt = 1 / FPS
        self.ui_time = 0.0
        self.menu_intro = 0.0
        self.menu_focus = [1.0, 0.0, 0.0]
        self.tutorial_transition = 1.0
        self.loadout_intro = 0.0
        self.weapon_intro = 0.0
        self.running = True
        self.state = "menu"
        self.menu_selected = 0
        self.tutorial_page = 0
        self.tutorial_step = 0
        self.tutorial_progress = 0.0
        self.tutorial_hits = 0
        self.arena_id = 1
        self.arena = Arena(self.arena_id)
        self.selected_weapon = WeaponType.PULSE
        self.session_upgrades: list[str] = []
        self.pending_arena_id = 2
        self.upgrade_choices = [UpgradeType.HEALTH, UpgradeType.DAMAGE, UpgradeType.SPEED]
        self.player = Player((180, HEIGHT / 2))
        self.player.set_weapon(self.selected_weapon)
        self.enemies: list[Enemy] = []
        self.projectiles: list[Projectile] = []
        self.effects: list[SlashEffect] = []
        self.hazards: list[TrapHazard | TurretHazard] = []
        self.decoys: list[Decoy] = []
        self.combat_fx = CombatEffects()
        self.audio = AudioManager()
        self.player_registry = PlayerRegistry.load(DATA_DIR / "players.json")
        self.current_player = self.player_registry.active
        self.player_select_index = max(0, self.player_registry.profiles.index(self.current_player))
        self.player_name_input = ""
        self.player_name_focused = False
        self.player_management_mode = "create"
        self.player_profile_notice = ""
        self.collector = GameplayDataCollector(SAMPLE_INTERVAL, self.current_player.player_id)
        self.data_quality: DataQualityReport = analyze_data_quality([])
        self.calibration_challenge = CalibrationChallenge(1)
        self.frame_outcome = ActionOutcome()
        self.session_registered = False
        self.profile = BehaviorProfile.from_samples([])
        self.external_model = action_model
        self.predictor = action_model or PlaceholderPredictor()
        self.model_metrics: TrainingMetrics | TorchTrainingMetrics | None = None
        self.ai_training_error = ""
        self.analysis_notice = ""
        self.neural_model_active = action_model is not None
        self.explicit_action: PlayerAction | None = None
        self.round_clear_timer = 0.0
        self.respawn_timer = 0.0
        self.notice = ""
        self.notice_timer = 0.0
        self.saved_paths: tuple | None = None
        self.twin_start_predictability = 0.0
        self.twin_action_changes = 0
        self.previous_twin_action: PlayerAction | None = None
        self.live_prediction: PlayerAction | None = None
        self.live_prediction_confidence = 0.0
        self.model_prediction_timer = 0.0
        self.prediction_confidences: list[float] = []
        self.round_time_left = TRAINING_ROUND_SECONDS
        self.wave_number = 0
        self.enemies_defeated = 0
        self.session_time = 0.0
        self.combo_tracker = ComboTracker()
        self.combo_notice = ""
        self.combo_notice_timer = 0.0
        self.heatmap = PositionHeatmap()
        self.heatmap_sample_timer = 0.0
        self.sync_history: list[bool] = []
        self.sync_sample_timer = 0.0
        self.synchronization = 0.0
        self.twin_intro_timer = 0.0
        self.cleanup_announced = False
        self.selected_abilities = [AbilityType.TRAP, AbilityType.WAVE, AbilityType.SHIELD]
        self.loadout_selection = set(self.selected_abilities)
        self.loadout_focus = [1.0 if ability in self.loadout_selection else 0.0 for ability in AbilityType]
        self.loadout_bar_progress = [0.0 for _ in AbilityType]
        self.ability_system = AbilitySystem(self.selected_abilities)
        self.objective: TrainingObjective | None = None
        self.perfect_dodges = 0
        self.slow_motion_timer = 0.0
        self.feint_cooldown = 0.0
        self.current_arena_elapsed = 0.0
        self.replay_recorder = ReplayRecorder()
        self.reset_training_arena = False
        self.previous_replay = ReplayRecorder.load(DATA_DIR / "last_replay.json")
        self.replay_arena = 1
        self.replay_time = 0.0
        self.replay_paused = False
        self.editor_path = DATA_DIR / "custom_arena.json"
        self.editor = ArenaEditor(ArenaLayout.load(self.editor_path))
        self.lab_sessions: list[SessionSummary] = []
        self.lab_index = 0
        self.tournament: TournamentMatch | None = None
        self.twin_phase = 1
        self.twin_ability_timer = 4.0
        self.wave_counted = False
        self.last_prediction_event = -10.0
        self.custom_mode = "elimination"
        self.custom_time_left = 45.0
        self.custom_hold_progress = 0.0
        self.demo_mode = False
        self.ai_training_thread: threading.Thread | None = None
        self.ai_training_result: tuple | None = None
        self.ai_training_progress: dict = {"stage": "подготовка", "epoch": 0, "total": 1}
        self.metrics_history: list[dict] = self._load_metrics_history()
        self.confusion_grid_rect = pygame.Rect(0, 0, 0, 0)
        self.confusion_error_cursor: dict[tuple[int, int], int] = {}
        self.replay_return_state = ""
        self.replay_error_label = ""
        self.analysis_replay_backup: ReplayRecorder | None = None

    def run(self) -> None:
        while self.running:
            real_dt = min(0.033, self.clock.tick(FPS) / 1000.0)
            self.frame_dt = real_dt
            self.ui_time += real_dt
            self.animated_background.update(real_dt)
            if self.state == "menu":
                self.menu_intro = min(1.0, self.menu_intro + real_dt * 2.2)
            if self.state == "tutorial":
                self.tutorial_transition = min(1.0, self.tutorial_transition + real_dt * 3.2)
            if self.state == "loadout":
                self._update_loadout_animation(real_dt)
            if self.state == "weapon_select":
                self.weapon_intro = min(1.0, self.weapon_intro + real_dt * 2.8)
            self.slow_motion_timer = max(0.0, self.slow_motion_timer - real_dt)
            dt = real_dt * (0.34 if self.slow_motion_timer > 0 else 1.0)
            self._handle_events()
            self._poll_ai_training()
            if self.state in ("training", "twin", "custom"):
                self._update_combat(dt)
            elif self.state == "tutorial_play":
                self._update_interactive_tutorial(dt)
            elif self.state == "replay":
                self._update_replay(dt)
            elif self.state == "tournament" and self.tournament:
                self.tournament.update(dt)
            self._draw()
            self._present_frame()
            pygame.display.flip()
        self._save_data()
        pygame.quit()

    def _refresh_viewport(self) -> None:
        self.viewport.update(self.display_surface.get_size())
        self._create_render_surfaces()

    def _create_render_surfaces(self) -> None:
        for name in ("screen", "world_surface", "alpha_surface"):
            previous = getattr(self, name, None)
            if previous is not None:
                unregister_surface(previous)
        render_size = self.viewport.rect.size
        render_scale = render_size[0] / WIDTH
        self.render_scale = render_scale
        renders_directly_to_display = self.viewport.rect.topleft == (0, 0) and render_size == self.display_surface.get_size()
        if renders_directly_to_display:
            self.screen = register_surface(self.display_surface, render_scale)
        else:
            self.screen = register_surface(pygame.Surface(render_size).convert(), render_scale)
        self.world_surface = register_surface(pygame.Surface(render_size).convert(), render_scale)
        self.alpha_surface = register_surface(
            pygame.Surface(render_size, pygame.SRCALPHA).convert_alpha(),
            render_scale,
        )

    def _toggle_fullscreen(self) -> None:
        if self.fullscreen:
            self.fullscreen = False
            self.display_surface = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        else:
            current_size = self.display_surface.get_size()
            if current_size[0] >= 800 and current_size[1] >= 600:
                self.windowed_size = current_size
            self.fullscreen = True
            desktop_sizes = pygame.display.get_desktop_sizes()
            self.desktop_size = desktop_sizes[0] if desktop_sizes else self.desktop_size
            self.display_surface = pygame.display.set_mode(self.desktop_size, pygame.FULLSCREEN | pygame.DOUBLEBUF)
        pygame.display.set_caption(TITLE)
        self._refresh_viewport()

    def _present_frame(self) -> None:
        if self.screen is self.display_surface:
            return
        self.display_surface.fill((3, 5, 10))
        self.display_surface.blit(self.screen, self.viewport.rect)

    def _mouse_position(self, *, clamp: bool = False) -> tuple[int, int]:
        return self.viewport.display_to_logical(pygame.mouse.get_pos(), clamp=clamp)

    @staticmethod
    def _clipboard_text() -> str:
        try:
            if not pygame.scrap.get_init():
                pygame.scrap.init()
            raw = pygame.scrap.get(pygame.SCRAP_TEXT)
        except pygame.error:
            return ""
        if raw is None:
            return ""
        if isinstance(raw, str):
            decoded = raw
        else:
            decoded = ""
            for encoding in ("utf-8", "utf-16-le", "mbcs", "cp1251"):
                try:
                    decoded = raw.decode(encoding)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
        cleaned = " ".join(decoded.replace("\x00", "").splitlines())
        return "".join(character for character in cleaned if character.isprintable())

    def _handle_events(self) -> None:
        self.explicit_action = None
        self.frame_outcome = ActionOutcome()
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.windowed_size = (max(800, event.w), max(600, event.h))
                self.display_surface = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
                self._refresh_viewport()
                continue
            if hasattr(event, "pos"):
                values = event.dict.copy()
                values["pos"] = self.viewport.display_to_logical(event.pos)
                event = pygame.event.Event(event.type, values)
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11 or (event.key == pygame.K_RETURN and getattr(event, "mod", 0) & pygame.KMOD_ALT):
                    self._toggle_fullscreen()
                    continue
                if event.key == pygame.K_ESCAPE:
                    if self.state == "menu":
                        self.running = False
                    elif self.state in ("editor", "lab", "tournament", "loadout", "weapon_select", "tutorial", "tutorial_play", "upgrade", "player_select"):
                        if self.state == "player_select":
                            pygame.key.stop_text_input()
                        self.state = "menu"
                    elif self.state == "model_analysis":
                        self.state = "profile"
                    elif self.state == "replay":
                        if self.replay_return_state:
                            self.state = self.replay_return_state
                            self.replay_return_state = ""
                            self.replay_error_label = ""
                            if self.analysis_replay_backup is not None:
                                self.replay_recorder = self.analysis_replay_backup
                                self.analysis_replay_backup = None
                        else:
                            self.state = "result" if self.profile.sample_count else "menu"
                    else:
                        self.state = "menu"
                elif self.state == "menu":
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self.menu_selected = (self.menu_selected - 1) % 3
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self.menu_selected = (self.menu_selected + 1) % 3
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self._activate_menu_option(self.menu_selected)
                    elif pygame.K_1 <= event.key <= pygame.K_3:
                        self.menu_selected = event.key - pygame.K_1
                        self._activate_menu_option(self.menu_selected)
                    elif event.key == pygame.K_e:
                        self._activate_menu_option(1)
                    elif event.key == pygame.K_l:
                        self._open_lab()
                    elif event.key == pygame.K_t:
                        self._start_tournament()
                    elif event.key == pygame.K_r and (self.replay_recorder.frames or self.previous_replay):
                        self._start_replay()
                    elif event.key == pygame.K_F9:
                        self.demo_mode = not self.demo_mode
                elif self.state == "player_select":
                    visible_count = max(1, len(self._visible_player_profiles()))
                    if event.key == pygame.K_UP:
                        self.player_select_index = (self.player_select_index - 1) % visible_count
                    elif event.key == pygame.K_DOWN:
                        self.player_select_index = (self.player_select_index + 1) % visible_count
                    elif event.key == pygame.K_v and getattr(event, "mod", 0) & pygame.KMOD_CTRL:
                        available = PLAYER_NAME_MAX_LENGTH - len(self.player_name_input)
                        if self.player_name_focused and available > 0:
                            self.player_name_input += self._clipboard_text()[:available]
                    elif event.key == pygame.K_BACKSPACE:
                        self.player_name_input = self.player_name_input[:-1]
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if self.player_management_mode == "delete_confirm":
                            self._request_or_confirm_player_delete()
                        else:
                            self._confirm_player_selection()
                elif self.state == "tutorial":
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        self._set_tutorial_page(self.tutorial_page - 1)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        self._set_tutorial_page(self.tutorial_page + 1)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if self.tutorial_page < 4:
                            self._set_tutorial_page(self.tutorial_page + 1)
                        else:
                            self._start_interactive_tutorial()
                elif self.state == "loadout":
                    if pygame.K_1 <= event.key <= pygame.K_7:
                        self._toggle_loadout_ability(event.key - pygame.K_1)
                    elif event.key == pygame.K_RETURN and len(self.loadout_selection) == 3:
                        self.selected_abilities = [ability for ability in AbilityType if ability in self.loadout_selection]
                        self._start_session()
                elif self.state == "weapon_select":
                    if pygame.K_1 <= event.key <= pygame.K_4:
                        self.selected_weapon = list(WeaponType)[event.key - pygame.K_1]
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self._open_loadout()
                elif self.state == "upgrade":
                    if pygame.K_1 <= event.key <= pygame.K_3:
                        self._choose_upgrade(event.key - pygame.K_1)
                elif self.state == "tutorial_play":
                    if self.tutorial_step == 4 and event.key == pygame.K_SPACE:
                        movement = self._movement_input()
                        if self.player.dash(movement, self.arena):
                            self.combat_fx.dash(self.player.position, COLORS["player"], movement.normalize() if movement.length_squared() else self.player.facing)
                            self._advance_tutorial_step()
                    elif self.tutorial_step == 5 and event.key == pygame.K_1:
                        self._activate_ability(0)
                        self._advance_tutorial_step()
                    elif self.tutorial_step >= 6 and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self.state = "menu"
                elif self.state == "profile":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self._start_twin_fight()
                    elif event.key == pygame.K_a:
                        self.state = "model_analysis"
                elif self.state == "result":
                    if event.key == pygame.K_r:
                        self._open_weapon_select()
                    elif event.key == pygame.K_v:
                        self._start_replay()
                    elif event.key == pygame.K_l:
                        self._open_lab()
                elif self.state == "editor":
                    if event.key == pygame.K_s:
                        self.editor.layout.save(self.editor_path)
                        self.editor.message = "Арена сохранена"
                    elif event.key == pygame.K_t:
                        self.editor.layout.save(self.editor_path)
                        self._start_custom_arena()
                    else:
                        self.editor.handle_event(event)
                elif self.state == "replay":
                    if event.key == pygame.K_SPACE:
                        self.replay_paused = not self.replay_paused
                    elif pygame.K_1 <= event.key <= pygame.K_4:
                        self.replay_arena = event.key - pygame.K_0
                        self.replay_time = 0.0
                elif self.state == "lab":
                    if event.key == pygame.K_LEFT:
                        self.lab_index = max(0, self.lab_index - 1)
                    elif event.key == pygame.K_RIGHT:
                        self.lab_index = min(max(0, len(self.lab_sessions) - 1), self.lab_index + 1)
                elif self.state == "tournament" and event.key == pygame.K_r:
                    self._start_tournament()
                elif self.state in ("training", "twin", "custom"):
                    if event.key == pygame.K_q:
                        health_before = self.player.health
                        if self.player.heal():
                            self.explicit_action = PlayerAction.HEAL
                            self.frame_outcome.effective_heal += self.player.health - health_before
                            if self.state == "training":
                                self.calibration_challenge.record("heal")
                    elif event.key == pygame.K_SPACE:
                        movement = self._movement_input()
                        perfect = self._is_imminent_hit()
                        if self.player.dash(movement, self.arena):
                            self.explicit_action = PlayerAction.DASH
                            if self.state == "training":
                                self.calibration_challenge.record("dash")
                            self.combat_fx.dash(self.player.position, COLORS["player"], movement.normalize() if movement.length_squared() else self.player.facing)
                            self.audio.play("dash")
                            if perfect:
                                self._trigger_perfect_dodge()
                    elif pygame.K_1 <= event.key <= pygame.K_3:
                        self._activate_ability(event.key - pygame.K_1)
                    elif event.key == pygame.K_f:
                        self._perform_feint()
            elif event.type == pygame.TEXTINPUT and self.state == "player_select" and self.player_name_focused:
                available = PLAYER_NAME_MAX_LENGTH - len(self.player_name_input)
                if available > 0:
                    self.player_name_input += "".join(character for character in event.text if character.isprintable())[:available]
            elif self.state == "menu" and event.type == pygame.MOUSEMOTION:
                for index, rect in enumerate(self._menu_button_rects()):
                    if rect.collidepoint(event.pos):
                        self.menu_selected = index
                        break
            elif self.state == "menu" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._profile_badge_rect().collidepoint(event.pos):
                    self._open_player_select()
                    continue
                for index, rect in enumerate(self._menu_button_rects()):
                    if rect.collidepoint(event.pos):
                        self.menu_selected = index
                        self._activate_menu_option(index)
                        break
            elif self.state == "player_select" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._player_rename_rect().collidepoint(event.pos):
                    self._begin_player_rename()
                    continue
                if self._player_delete_rect().collidepoint(event.pos):
                    self._request_or_confirm_player_delete()
                    continue
                if self._player_name_input_rect().collidepoint(event.pos):
                    self.player_name_focused = True
                    if self.player_management_mode != "rename":
                        self.player_management_mode = "create"
                        self.player_profile_notice = ""
                    pygame.key.start_text_input()
                for index, rect in enumerate(self._player_card_rects()):
                    if rect.collidepoint(event.pos):
                        self.player_select_index = index
                        self.player_name_input = ""
                        self.player_name_focused = False
                        self.player_management_mode = "create"
                        self.player_profile_notice = ""
                        pygame.key.stop_text_input()
                        break
                if self._player_continue_rect().collidepoint(event.pos):
                    self._confirm_player_selection()
            elif self.state == "profile" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for index, rect in enumerate(self._profile_action_rects()):
                    if rect.collidepoint(event.pos):
                        self._activate_profile_option(index)
                        break
                else:
                    self._open_confusion_example(event.pos)
            elif self.state == "result" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for index, rect in enumerate(self._result_button_rects()):
                    if rect.collidepoint(event.pos):
                        self._activate_result_option(index)
                        break
            elif self.state == "tutorial" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                back, forward = self._tutorial_navigation_rects()
                if back.collidepoint(event.pos):
                    if self.tutorial_page == 0:
                        self.state = "menu"
                    else:
                        self._set_tutorial_page(self.tutorial_page - 1)
                elif forward.collidepoint(event.pos):
                    if self.tutorial_page == 4:
                        self._start_interactive_tutorial()
                    else:
                        self._set_tutorial_page(self.tutorial_page + 1)
            elif self.state == "loadout" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._loadout_continue_rect().collidepoint(event.pos) and len(self.loadout_selection) == 3:
                    self.selected_abilities = [ability for ability in AbilityType if ability in self.loadout_selection]
                    self._start_session()
                    continue
                for index, rect in enumerate(self._loadout_card_rects()):
                    if rect.collidepoint(event.pos):
                        self._toggle_loadout_ability(index)
                        break
            elif self.state == "weapon_select" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._weapon_continue_rect().collidepoint(event.pos):
                    self._open_loadout()
                    continue
                for index, rect in enumerate(self._weapon_card_rects()):
                    if rect.collidepoint(event.pos):
                        self.selected_weapon = list(WeaponType)[index]
                        break
            elif self.state == "upgrade" and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for index, rect in enumerate(self._upgrade_card_rects()):
                    if rect.collidepoint(event.pos):
                        self._choose_upgrade(index)
                        break
            elif self.state == "editor":
                self.editor.handle_event(event)

    @staticmethod
    def _menu_button_rects() -> list[pygame.Rect]:
        return [pygame.Rect(WIDTH // 2 - 260, 335 + index * 82, 520, 62) for index in range(3)]

    @staticmethod
    def _tutorial_navigation_rects() -> tuple[pygame.Rect, pygame.Rect]:
        return pygame.Rect(105, 635, 210, 52), pygame.Rect(WIDTH - 315, 635, 210, 52)

    @staticmethod
    def _result_button_rects() -> list[pygame.Rect]:
        return [pygame.Rect(70 + index * 290, 575, 270, 46) for index in range(4)]

    @staticmethod
    def _profile_action_rects() -> list[pygame.Rect]:
        return [pygame.Rect(205 + index * 450, 638, 420, 44) for index in range(2)]

    def _activate_profile_option(self, index: int) -> None:
        if index == 0:
            self._start_twin_fight()
        elif index == 1:
            self.state = "model_analysis"

    def _activate_result_option(self, index: int) -> None:
        if index == 0:
            self._open_weapon_select()
        elif index == 1:
            self._start_replay()
        elif index == 2:
            self._open_lab()
        elif index == 3:
            self.state = "menu"

    def _activate_menu_option(self, index: int) -> None:
        if index == 0:
            self._open_player_select()
        elif index == 1:
            self.state = "editor"
        else:
            self.tutorial_page = 0
            self.tutorial_transition = 0.0
            self.state = "tutorial"

    def _open_player_select(self) -> None:
        self.player_registry = PlayerRegistry.load(DATA_DIR / "players.json")
        self.current_player = self.player_registry.active
        self.player_select_index = self._visible_player_profiles().index(self.current_player)
        self.player_name_input = ""
        self.player_name_focused = True
        self.player_management_mode = "create"
        self.player_profile_notice = ""
        pygame.key.start_text_input()
        self.state = "player_select"

    def _visible_player_profiles(self) -> list[PlayerProfileRecord]:
        profiles = self.player_registry.profiles
        if self.current_player in profiles[:5]:
            return profiles[:5]
        return [self.current_player, *[profile for profile in profiles if profile != self.current_player][:4]]

    @staticmethod
    def _player_continue_rect() -> pygame.Rect:
        return pygame.Rect(WIDTH // 2 - 205, 620, 410, 48)

    @staticmethod
    def _player_name_input_rect() -> pygame.Rect:
        return pygame.Rect(330, 570, 620, 42)

    @staticmethod
    def _profile_badge_rect() -> pygame.Rect:
        return pygame.Rect(WIDTH // 2 - 170, 246, 340, 42)

    @staticmethod
    def _player_rename_rect() -> pygame.Rect:
        return pygame.Rect(700, 130, 160, 34)

    @staticmethod
    def _player_delete_rect() -> pygame.Rect:
        return pygame.Rect(875, 130, 160, 34)

    def _player_card_rects(self) -> list[pygame.Rect]:
        visible = self._visible_player_profiles()
        return [pygame.Rect(245, 175 + index * 76, 790, 60) for index in range(len(visible))]

    def _selected_player_profile(self) -> PlayerProfileRecord:
        profiles = self._visible_player_profiles()
        self.player_select_index = max(0, min(self.player_select_index, len(profiles) - 1))
        return profiles[self.player_select_index]

    def _begin_player_rename(self) -> None:
        profile = self._selected_player_profile()
        self.player_management_mode = "rename"
        self.player_name_input = profile.name
        self.player_name_focused = True
        self.player_profile_notice = "Введите новое имя и нажмите «Сохранить имя»"
        pygame.key.start_text_input()

    def _request_or_confirm_player_delete(self) -> None:
        profile = self._selected_player_profile()
        if len(self.player_registry.profiles) <= 1:
            self.player_profile_notice = "Нельзя удалить единственный профиль"
            return
        if self.player_management_mode != "delete_confirm":
            self.player_management_mode = "delete_confirm"
            self.player_name_input = ""
            self.player_name_focused = False
            self.player_profile_notice = f"Удалить профиль «{profile.name}»? Нажмите удаление ещё раз"
            pygame.key.stop_text_input()
            return
        self.current_player = self.player_registry.delete(profile.player_id)
        self.player_select_index = self._visible_player_profiles().index(self.current_player)
        self.player_management_mode = "create"
        self.player_profile_notice = "Профиль удалён. Собранные игровые данные сохранены"

    def _confirm_player_selection(self) -> None:
        if self.player_management_mode == "rename":
            try:
                profile = self.player_registry.rename(self._selected_player_profile().player_id, self.player_name_input)
            except ValueError as error:
                self.player_profile_notice = str(error)
                return
            if profile.player_id == self.current_player.player_id:
                self.current_player = profile
            self.player_name_input = ""
            self.player_name_focused = False
            self.player_management_mode = "create"
            self.player_profile_notice = "Имя профиля сохранено"
            pygame.key.stop_text_input()
            return
        if self.player_name_input.strip():
            self.current_player = self.player_registry.create(self.player_name_input)
        else:
            profiles = self._visible_player_profiles()
            self.player_select_index = max(0, min(self.player_select_index, len(profiles) - 1))
            self.current_player = self.player_registry.select(profiles[self.player_select_index].player_id)
        self.player_name_input = ""
        self.player_name_focused = False
        pygame.key.stop_text_input()
        self._open_weapon_select()

    def _open_loadout(self) -> None:
        self.loadout_intro = 0.0
        self.loadout_bar_progress = [0.0 for _ in AbilityType]
        self.state = "loadout"

    def _open_weapon_select(self) -> None:
        self.weapon_intro = 0.0
        self.state = "weapon_select"

    @staticmethod
    def _weapon_card_rects() -> list[pygame.Rect]:
        return [pygame.Rect(130 + index % 2 * 520, 205 + index // 2 * 160, 490, 125) for index in range(len(WeaponType))]

    @staticmethod
    def _weapon_continue_rect() -> pygame.Rect:
        return pygame.Rect(WIDTH // 2 - 260, 610, 520, 48)

    @staticmethod
    def _upgrade_card_rects() -> list[pygame.Rect]:
        return [pygame.Rect(165 + index * 330, 270, 290, 220) for index in range(3)]

    def _choose_upgrade(self, index: int) -> None:
        if index < 0 or index >= len(self.upgrade_choices):
            return
        chosen = self.upgrade_choices[index]
        self.session_upgrades.append(chosen.value)
        self.arena_id = self.pending_arena_id
        self.state = "training"
        self._setup_arena()
        self.notice = f"УЛУЧШЕНИЕ: {UPGRADE_INFO[chosen][0].upper()}"
        self.notice_timer = 2.2

    def _update_loadout_animation(self, dt: float) -> None:
        self.loadout_intro = min(1.0, self.loadout_intro + dt * 2.8)
        for index, ability in enumerate(AbilityType):
            if ability in self.loadout_selection:
                self.loadout_bar_progress[index] = min(1.0, self.loadout_bar_progress[index] + dt * 0.72)
            else:
                self.loadout_bar_progress[index] = 0.0

    @staticmethod
    def _loadout_card_rects() -> list[pygame.Rect]:
        return [pygame.Rect(145 + index % 2 * 510, 150 + index // 2 * 112, 470, 88) for index in range(len(AbilityType))]

    @staticmethod
    def _loadout_continue_rect() -> pygame.Rect:
        return pygame.Rect(WIDTH // 2 - 220, 650, 440, 48)

    def _toggle_loadout_ability(self, index: int) -> None:
        ability = list(AbilityType)[index]
        changed = False
        if ability in self.loadout_selection:
            self.loadout_selection.remove(ability)
            changed = True
        elif len(self.loadout_selection) < 3:
            self.loadout_selection.add(ability)
            changed = True
        if changed:
            self.loadout_bar_progress = [0.0 for _ in AbilityType]

    def _set_tutorial_page(self, page: int) -> None:
        page = max(0, min(4, page))
        if page != self.tutorial_page:
            self.tutorial_page = page
            self.tutorial_transition = 0.0

    def _start_interactive_tutorial(self) -> None:
        self.state = "tutorial_play"
        self.tutorial_step = 0
        self.tutorial_progress = 0.0
        self.tutorial_hits = 0
        self.arena = Arena(1)
        self.arena.destructibles = []
        self.player = Player((self.arena.bounds.left + 115, self.arena.bounds.centery))
        self.player.set_weapon(WeaponType.PULSE)
        self.enemies = [Enemy((self.arena.bounds.right - 215, self.arena.bounds.centery), "assault")]
        self.enemies[0].max_health = 9999
        self.enemies[0].health = 9999
        self.enemies[0].attack_cooldown = 9999
        self.projectiles = []
        self.effects = []
        self.hazards = []
        self.decoys = []
        self.combat_fx = CombatEffects()
        self.ability_system = AbilitySystem([AbilityType.WAVE, AbilityType.SHIELD, AbilityType.TELEPORT])
        self.notice = "ПРАКТИЧЕСКОЕ ОБУЧЕНИЕ"
        self.notice_timer = 1.8

    def _advance_tutorial_step(self) -> None:
        self.tutorial_step += 1
        self.tutorial_progress = 0.0
        self.projectiles.clear()
        self.player.energy = self.player.max_energy
        if self.tutorial_step < len(INTERACTIVE_TUTORIAL_STEPS):
            self.notice = f"ЭТАП {self.tutorial_step + 1}/{len(INTERACTIVE_TUTORIAL_STEPS)}"
            self.notice_timer = 1.0
        else:
            self.notice = "ОБУЧЕНИЕ ЗАВЕРШЕНО"
            self.notice_timer = 999.0

    def _update_interactive_tutorial(self, dt: float) -> None:
        self.notice_timer = max(0.0, self.notice_timer - dt)
        self.combat_fx.update(dt)
        self.effects = [effect for effect in self.effects if effect.update(dt)]
        if self.tutorial_step >= len(INTERACTIVE_TUTORIAL_STEPS):
            return
        movement = self._movement_input()
        blocking = bool(pygame.key.get_pressed()[pygame.K_LSHIFT])
        self.player.set_block(blocking, dt)
        self.player.update(dt, movement, self.arena)
        if self.tutorial_step == 0 and movement.length_squared() > 0:
            self.tutorial_progress = min(1.5, self.tutorial_progress + dt)
            if self.tutorial_progress >= 1.5:
                self._advance_tutorial_step()
                return
        elif self.tutorial_step == 3 and blocking:
            self.tutorial_progress = min(1.0, self.tutorial_progress + dt)
            if self.tutorial_progress >= 1.0:
                self._advance_tutorial_step()
                return

        target = self.enemies[0]
        mouse_position = pygame.Vector2(self._mouse_position(clamp=True))
        mouse_buttons = pygame.mouse.get_pressed(3)
        if self.tutorial_step == 1 and mouse_buttons[0]:
            fired = self.player.fire(mouse_position)
            if fired:
                self.projectiles.extend(fired)
                self.audio.play("shot")
        elif self.tutorial_step == 2 and mouse_buttons[2]:
            hit, effect = self.player.melee(mouse_position, [target], self.arena.has_line_of_sight)
            if effect:
                self.effects.append(effect)
            if hit:
                self.combat_fx.hit(target.position, COLORS["player"], 1, strong=True)
                self._advance_tutorial_step()
                return

        active_projectiles: list[Projectile] = []
        for projectile in self.projectiles:
            if not projectile.update(dt, self.arena):
                if projectile.impact_position is not None:
                    self.combat_fx.hit(projectile.impact_position, COLORS["warning"], projectile.damage, strong=True)
                continue
            if projectile.position.distance_to(target.position) <= target.radius + projectile.radius:
                self.tutorial_hits += 1
                self.combat_fx.hit(target.position, COLORS["player"], 1)
                if self.tutorial_hits >= 3 and self.tutorial_step == 1:
                    self._advance_tutorial_step()
                    return
            else:
                active_projectiles.append(projectile)
        self.projectiles = active_projectiles

    def _create_player(self, position: tuple[float, float]) -> Player:
        player = Player(position)
        player.set_weapon(self.selected_weapon)
        for upgrade in self.session_upgrades:
            if upgrade == "health":
                player.max_health += 20
                player.health = player.max_health
            elif upgrade == "energy":
                player.max_energy += 20
                player.energy = player.max_energy
            elif upgrade == "speed":
                player.speed_multiplier *= 1.12
            elif upgrade == "damage":
                player.damage_multiplier *= 1.15
            elif upgrade == "cooldown":
                player.cooldown_multiplier *= 0.86
            elif upgrade == "heal":
                player.heal_charges += 1
        return player

    def _start_session(self) -> None:
        self.collector = GameplayDataCollector(SAMPLE_INTERVAL, self.current_player.player_id)
        self.profile = BehaviorProfile.from_samples([])
        self.saved_paths = None
        self.model_metrics = None
        self.ai_training_error = ""
        self.analysis_notice = ""
        self.data_quality = analyze_data_quality([])
        self.session_registered = False
        self.neural_model_active = self.external_model is not None
        self.combo_tracker = ComboTracker()
        self.heatmap = PositionHeatmap()
        self.enemies_defeated = 0
        self.perfect_dodges = 0
        self.session_time = 0.0
        self.sync_history.clear()
        self.synchronization = 0.0
        self.ability_system = AbilitySystem(self.selected_abilities)
        self.replay_recorder = ReplayRecorder()
        self.last_prediction_event = -10.0
        self.session_upgrades = []
        self.arena_id = 1
        self.state = "training"
        self._setup_arena()
        self.notice = "КАЛИБРОВКА 1/3: удержание зоны"
        self.notice_timer = 2.8

    def _setup_arena(self) -> None:
        self.arena = Arena(self.arena_id)
        self.player = self._create_player((self.arena.bounds.left + 105, self.arena.bounds.centery))
        self.projectiles.clear()
        self.effects.clear()
        self.hazards.clear()
        self.decoys.clear()
        self.combat_fx = CombatEffects()
        self.round_clear_timer = 0.0
        self.wave_counted = False
        self.reset_training_arena = False
        self.respawn_timer = 0.0
        self.current_arena_elapsed = 0.0
        if self.arena_id <= 3:
            self.round_time_left = 14.0 if self.demo_mode else TRAINING_ROUND_SECONDS
            self.wave_number = 0
            self.cleanup_announced = False
            self.enemies = []
            self.objective = TrainingObjective(self.arena_id, self.arena.bounds)
            self.calibration_challenge = CalibrationChallenge(self.arena_id)
            if self.arena_id == 3:
                self.player.health = min(self.player.health, self.player.max_health * 0.62)
                self.player.heal_charges = max(2, self.player.heal_charges)
            self._spawn_training_wave()
        else:
            self.objective = None
            self.enemies = [Enemy((self.arena.bounds.right - 155, self.arena.bounds.centery), "twin", self.profile)]
            self.predictor.reset_history()
            self.model_prediction_timer = 0.0

    def _spawn_training_wave(self) -> None:
        self.wave_number += 1
        right = self.arena.bounds.right
        top = self.arena.bounds.top
        bottom = self.arena.bounds.bottom
        center = self.arena.bounds.centery
        if self.arena_id == 1:
            kinds = ["assault"] if self.wave_number % 2 else ["assault", "shield"]
        elif self.arena_id == 2:
            kinds = ["sniper", "teleporter"] if self.wave_number % 2 else ["sniper", "shield"]
        else:
            kinds = ["engineer", "copier"] if self.wave_number % 2 else ["assault", "engineer", "copier"]
        positions = [(right - 130, top + 130), (right - 145, bottom - 120), (right - 270, center)]
        self.enemies = [Enemy(positions[index], kind) for index, kind in enumerate(kinds)]
        self.projectiles.clear()
        self.player.health = min(self.player.max_health, self.player.health + 16)
        self.player.energy = min(self.player.max_energy, self.player.energy + 28)
        self.round_clear_timer = 0.0
        self.wave_counted = False
        self.notice = f"ВОЛНА {self.wave_number}"
        self.notice_timer = 1.0

    def _start_twin_fight(self) -> None:
        self.arena_id = 4
        self.state = "twin"
        self._setup_arena()
        self.twin_start_predictability = self.profile.predictability
        self.twin_action_changes = 0
        self.previous_twin_action = None
        self.live_prediction = None
        self.live_prediction_confidence = 0.0
        self.model_prediction_timer = 0.0
        self.prediction_confidences.clear()
        self.predictor.reset_history()
        self.sync_history.clear()
        self.synchronization = self.profile.predictability
        self.twin_intro_timer = TWIN_INTRO_SECONDS
        self.twin_phase = 1
        self.twin_ability_timer = 4.0
        self.notice = "СИНХРОНИЗАЦИЯ ЗАВЕРШЕНА — ДВОЙНИК АКТИВЕН"
        self.notice_timer = 3.2

    def _start_custom_arena(self) -> None:
        layout = self.editor.layout
        self.state = "custom"
        self.arena_id = 5
        self.arena = Arena.from_layout(layout)
        self.player = self._create_player(layout.player_spawn)
        self.player.set_weapon(self.editor.player_weapon)
        self.player.position = self.arena.place_circle(self.player.position, self.player.radius)
        self.enemies = self._create_custom_enemies()
        self.projectiles = []
        self.effects = []
        self.hazards = []
        self.decoys = []
        self.objective = None
        self.ability_system = AbilitySystem(self.selected_abilities)
        self.current_arena_elapsed = 0.0
        self.round_clear_timer = 0.0
        self.custom_mode = layout.game_mode
        self.custom_time_left = 45.0
        self.custom_hold_progress = 0.0
        self.notice = f"РЕЖИМ: {GAME_MODE_NAMES.get(self.custom_mode, 'Уничтожение').upper()}"
        self.notice_timer = 2.0

    def _open_lab(self) -> None:
        self.lab_sessions = load_sessions(DATA_DIR)
        self.lab_index = min(self.lab_index, max(0, len(self.lab_sessions) - 1))
        self.state = "lab"

    def _start_tournament(self) -> None:
        sessions = load_sessions(DATA_DIR)
        if len(sessions) >= 2:
            first = TournamentFighter.from_session(sessions[0], 1)
            second = TournamentFighter.from_session(sessions[1], 2)
        else:
            first = TournamentFighter.demo("Двойник 1: Штурм", True)
            second = TournamentFighter.demo("Двойник 2: Тактика", False)
        self.tournament = TournamentMatch(first, second)
        self.state = "tournament"

    def _start_replay(self) -> None:
        if not self.replay_recorder.frames and self.previous_replay:
            self.replay_recorder = self.previous_replay
        self.replay_arena = 1
        self.replay_time = 0.0
        self.replay_paused = False
        self.replay_return_state = ""
        self.replay_error_label = ""
        self.state = "replay"

    def _update_replay(self, dt: float) -> None:
        if self.replay_paused:
            return
        frames = self.replay_recorder.frames_for_arena(self.replay_arena)
        if not frames:
            return
        self.replay_time += dt
        maximum = max(frame.arena_time for frame in frames)
        if self.replay_time > maximum:
            self.replay_time = 0.0

    def _is_imminent_hit(self) -> bool:
        for enemy in self.enemies:
            if enemy.alive and 0 < enemy.attack_windup <= 0.22:
                if enemy.queued_attack == "melee" and enemy.position.distance_to(self.player.position) < 95:
                    return True
                if enemy.queued_attack == "ranged":
                    return True
        for projectile in self.projectiles:
            if projectile.owner != "enemy":
                continue
            to_player = self.player.position - projectile.position
            if to_player.length() < 115 and projectile.velocity.dot(to_player) > 0:
                return True
        return False

    def _trigger_perfect_dodge(self) -> None:
        self.perfect_dodges += 1
        self.frame_outcome.perfect_dodge += 1
        if self.state == "training":
            self.calibration_challenge.record("perfect_dodge")
        self.player.energy = min(self.player.max_energy, self.player.energy + 32)
        self.slow_motion_timer = 0.48
        self.notice = "ИДЕАЛЬНОЕ УКЛОНЕНИЕ  +32 энергии"
        self.notice_timer = 1.1
        self.replay_recorder.add_event(self.arena_id, self.current_arena_elapsed, "Идеальное уклонение")
        self.audio.play("perfect")

    def _perform_feint(self) -> None:
        if self.feint_cooldown > 0:
            return
        self.feint_cooldown = 2.2
        self.explicit_action = PlayerAction.RANGED_ATTACK
        self.player.last_action = PlayerAction.RANGED_ATTACK
        self.combat_fx.wave(self.player.position, COLORS["muted"])
        self.notice = "ЛОЖНАЯ АТАКА — ПРОФИЛЬ ПОЛУЧИЛ ЛОЖНЫЙ СИГНАЛ"
        self.notice_timer = 1.2
        self.replay_recorder.add_event(self.arena_id, self.current_arena_elapsed, "Ложная атака")

    def _activate_ability(self, slot: int) -> None:
        ability = self.ability_system.activate(slot, locked=self.player.ability_lock_timer > 0)
        if ability is None:
            return
        self.player.last_used_ability = ability.value
        mouse = pygame.Vector2(self._mouse_position(clamp=True))
        if ability == AbilityType.TRAP:
            self.hazards.append(TrapHazard(self.player.position.copy(), "player"))
        elif ability == AbilityType.WAVE:
            for enemy in self.enemies:
                if enemy.alive and enemy.position.distance_to(self.player.position) < 185:
                    was_alive = enemy.alive
                    applied = enemy.take_damage(28, self.player.position)
                    self.frame_outcome.damage_dealt += applied
                    self.frame_outcome.kill += int(was_alive and not enemy.alive)
                    self.combat_fx.hit(enemy.position, COLORS["player"], applied, strong=True)
            self.combat_fx.wave(self.player.position, COLORS["player"])
        elif ability == AbilityType.REFLECT:
            self.player.reflect_timer = 1.6
        elif ability == AbilityType.TELEPORT:
            direction = (mouse - self.player.position).normalize() if mouse.distance_squared_to(self.player.position) > 1 else self.player.facing
            old_position = self.player.position.copy()
            self.player.position = self.arena.place_circle(self.player.position + direction * 235, self.player.radius)
            self.combat_fx.dash(old_position, COLORS["twin"], direction)
        elif ability == AbilityType.SHIELD:
            self.player.shield_timer = 2.8
        elif ability == AbilityType.DECOY:
            self.decoys.append(Decoy(self.player.position.copy()))
        elif ability == AbilityType.SLOW:
            for enemy in self.enemies:
                enemy.slow_timer = max(enemy.slow_timer, 3.5)
        self.notice = f"СПОСОБНОСТЬ: {ABILITY_INFO[ability][0]}"
        self.notice_timer = 1.0
        self.replay_recorder.add_event(self.arena_id, self.current_arena_elapsed, ABILITY_INFO[ability][0])
        self.audio.play("ability")

    def _movement_input(self) -> pygame.Vector2:
        keys = pygame.key.get_pressed()
        return pygame.Vector2(
            float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
            float(keys[pygame.K_s]) - float(keys[pygame.K_w]),
        )

    def _nearest_enemy(self) -> Enemy | None:
        living = [enemy for enemy in self.enemies if enemy.alive]
        if not living:
            return None
        return min(living, key=lambda enemy: enemy.position.distance_squared_to(self.player.position))

    def _update_combat(self, dt: float) -> None:
        if self.state not in ("training", "twin", "custom"):
            return
        self.session_time += dt
        self.current_arena_elapsed += dt
        self.notice_timer = max(0.0, self.notice_timer - dt)
        self.combo_notice_timer = max(0.0, self.combo_notice_timer - dt)
        self.feint_cooldown = max(0.0, self.feint_cooldown - dt)
        self.model_prediction_timer = max(0.0, self.model_prediction_timer - dt)
        self.ability_system.update(dt)
        self.combat_fx.update(dt)
        self.decoys = [decoy for decoy in self.decoys if decoy.update(dt)]
        if self.state == "twin" and self.twin_intro_timer > 0:
            self.twin_intro_timer = max(0.0, self.twin_intro_timer - dt)
            return
        if self.respawn_timer > 0:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0:
                if self.state == "training":
                    if self.reset_training_arena:
                        self.reset_training_arena = False
                        self._setup_arena()
                    else:
                        self.player = self._create_player((self.arena.bounds.left + 105, self.arena.bounds.centery))
                        self.projectiles.clear()
                elif self.state == "twin":
                    self._setup_arena()
                    self.twin_intro_timer = 1.4
                else:
                    self._start_custom_arena()
                self.notice = "ПОВТОРНАЯ СИНХРОНИЗАЦИЯ"
                self.notice_timer = 1.4
            return

        if self.state == "training":
            self.round_time_left = max(0.0, self.round_time_left - dt)
            if self.round_time_left <= 0 and not self.cleanup_announced:
                self.cleanup_announced = True
                self.notice = "ВРЕМЯ ВЫШЛО — ЗАЧИСТИТЕ ОСТАВШИХСЯ"
                self.notice_timer = 2.5

        movement = self._movement_input()
        self.player.set_block(bool(pygame.key.get_pressed()[pygame.K_LSHIFT]), dt)
        self.player.update(dt, movement, self.arena)
        if self.state == "custom":
            if self.custom_mode == "survival":
                self.custom_time_left = max(0.0, self.custom_time_left - dt)
                if self.custom_time_left <= 0:
                    self._complete_custom_arena("Испытание на выживание пройдено")
                    return
            elif self.custom_mode == "hold":
                objective = pygame.Vector2(self.editor.layout.objective_position)
                if self.player.position.distance_to(objective) <= 90:
                    self.custom_hold_progress = min(15.0, self.custom_hold_progress + dt)
                if self.custom_hold_progress >= 15.0:
                    self._complete_custom_arena("Точка успешно удержана")
                    return
        mouse_position = pygame.Vector2(self._mouse_position(clamp=True))
        mouse_buttons = pygame.mouse.get_pressed(3)
        if mouse_buttons[0]:
            fired_projectiles = self.player.fire(mouse_position)
            if fired_projectiles:
                self.projectiles.extend(fired_projectiles)
                self.explicit_action = PlayerAction.RANGED_ATTACK
                self.audio.play("shot")
        if mouse_buttons[2]:
            health_before = {id(enemy): enemy.health for enemy in self.enemies if enemy.alive}
            used, effect = self.player.melee(
                mouse_position,
                [enemy for enemy in self.enemies if enemy.alive],
                self.arena.has_line_of_sight,
            )
            if effect:
                self.effects.append(effect)
                self.combat_fx.wave(self.player.position, COLORS["player"])
                weapon = WEAPON_SPECS[self.player.weapon]
                object_hit = self.arena.damage_destructible_near(
                    self.player.position,
                    weapon.melee_range,
                    weapon.melee_damage * self.player.damage_multiplier,
                )
                if object_hit is not None:
                    self.combat_fx.hit(object_hit, COLORS["warning"], weapon.melee_damage, strong=True)
                for enemy in self.enemies:
                    damage = health_before.get(id(enemy), enemy.health) - enemy.health
                    if damage > 0:
                        self.frame_outcome.damage_dealt += damage
                        self.frame_outcome.kill += int(health_before.get(id(enemy), 0) > 0 and not enemy.alive)
                        if self.state == "training":
                            self.calibration_challenge.record("melee_hit")
                        self.combat_fx.hit(enemy.position, COLORS["player"], damage, strong=True)
                        self.audio.play("hit")
            if used or effect:
                self.explicit_action = PlayerAction.MELEE_ATTACK

        for pack in self.arena.health_packs:
            if pack.active and pack.position.distance_to(self.player.position) < 36:
                pack.active = False
                self.player.heal_charges = min(2, self.player.heal_charges + 1)
                self.notice = "+1 ЗАРЯД ЛЕЧЕНИЯ [Q]"
                self.notice_timer = 1.2

        forced_target = self.decoys[-1].position if self.decoys else None
        twin = next((enemy for enemy in self.enemies if enemy.alive and enemy.kind == "twin"), None)
        if twin:
            ratio = twin.health / twin.max_health
            new_phase = 1 if ratio > 0.67 else (2 if ratio > 0.34 else 3)
            if new_phase != self.twin_phase:
                self.twin_phase = new_phase
                names = {1: "КОПИРОВАНИЕ", 2: "ПРЕДСКАЗАНИЕ", 3: "ИСКАЖЕНИЕ"}
                self.notice = f"ФАЗА {new_phase}: {names[new_phase]}"
                self.notice_timer = 2.0
                self.combat_fx.wave(twin.position, twin.color)
            self._update_twin_abilities(dt, twin)

        for enemy in self.enemies:
            if not enemy.alive:
                continue
            if enemy.kind == "copier":
                self._update_copier_ability(enemy)
            predicted_action = None
            if self.state == "twin" and enemy.kind == "twin":
                if self.model_prediction_timer <= 0.0 or self.live_prediction is None:
                    prediction_features = build_feature_vector(self.player, enemy, self.arena)
                    model_prediction = self.predictor.predict(prediction_features)
                    self.live_prediction = model_prediction.action
                    self.live_prediction_confidence = max(model_prediction.probabilities)
                    self.model_prediction_timer = SAMPLE_INTERVAL
                predicted_action = self.live_prediction
            previous_health = self.player.health
            enemy.update(
                dt,
                self.player,
                self.arena,
                self.projectiles,
                predicted_action,
                self.synchronization,
                self.hazards,
                forced_target,
                self.twin_phase,
            )
            if self.player.health < previous_health:
                damage = previous_health - self.player.health
                self.frame_outcome.damage_taken += damage
                if self.player.block_active:
                    self.frame_outcome.successful_block += 1
                    if self.state == "training":
                        self.calibration_challenge.record("block")
                self.combat_fx.hit(self.player.position, COLORS["enemy"], damage, strong=damage >= 18)
                self.audio.play("hit")
                self.replay_recorder.add_event(self.arena_id, self.current_arena_elapsed, f"Получен урон: {damage:.0f}")

        active_hazards: list[TrapHazard | TurretHazard] = []
        for hazard in self.hazards:
            if isinstance(hazard, TurretHazard):
                if hazard.update(dt, self.player, self.projectiles):
                    active_hazards.append(hazard)
            else:
                player_health_before = self.player.health
                enemy_health_before = {id(enemy): enemy.health for enemy in self.enemies if enemy.alive}
                active, hit_position, damage = hazard.update(dt, self.player, self.enemies)
                if hit_position is not None:
                    color = COLORS["player"] if hazard.owner == "player" else COLORS["enemy"]
                    self.combat_fx.hit(hit_position, color, damage, strong=True)
                    if hazard.owner == "player":
                        dealt = sum(max(0.0, enemy_health_before.get(id(enemy), enemy.health) - enemy.health) for enemy in self.enemies)
                        kills = sum(1 for enemy in self.enemies if enemy_health_before.get(id(enemy), 0) > 0 and not enemy.alive)
                        self.frame_outcome.damage_dealt += dealt
                        self.frame_outcome.kill += kills
                    else:
                        applied_damage = max(0.0, player_health_before - self.player.health)
                        self.frame_outcome.damage_taken += applied_damage
                        if self.player.block_active:
                            self.frame_outcome.successful_block += 1
                            if self.state == "training":
                                self.calibration_challenge.record("block")
                        self.replay_recorder.add_event(self.arena_id, self.current_arena_elapsed, f"Урон ловушки: {applied_damage:.0f}")
                if active:
                    active_hazards.append(hazard)
        self.hazards = active_hazards
        self._update_projectiles(dt)
        self.effects = [effect for effect in self.effects if effect.update(dt)]

        if self.objective:
            objective_before = self.objective.progress
            self.objective.update(dt, self.player.position, self.enemies)
            self.frame_outcome.objective_progress += max(0.0, self.objective.progress - objective_before)
            if self.objective.failed and self.state == "training":
                self.notice = "ОБЪЕКТ УНИЧТОЖЕН — ПОВТОР АРЕНЫ"
                self.notice_timer = 2.0
                self.reset_training_arena = True
                self.respawn_timer = 1.5

        nearest = self._nearest_enemy()
        training_features = (
            build_feature_vector(self.player, nearest, self.arena)
            if self.state == "training" and nearest is not None
            else None
        )
        current_action = classify_movement_action(self.player, nearest, self.explicit_action)
        self.player.last_action = current_action
        combo = self.combo_tracker.record(self.session_time, current_action)
        if combo:
            self.player.energy = min(self.player.max_energy, self.player.energy + combo.energy_bonus)
            self.combo_notice = f"{combo.name}  +{combo.energy_bonus:.0f} энергии"
            self.combo_notice_timer = 1.8
            self.replay_recorder.add_event(self.arena_id, self.current_arena_elapsed, combo.name)
            self.audio.play("combo")

        self.heatmap_sample_timer += dt
        if self.heatmap_sample_timer >= 0.1:
            self.heatmap_sample_timer %= 0.1
            self.heatmap.add(self.player.position, self.arena.bounds)
        if training_features is not None:
            self.collector.update(
                dt,
                self.arena_id,
                training_features,
                current_action,
                force=self.explicit_action is not None,
                arena_time=self.current_arena_elapsed,
            )
            self.collector.record_outcome(current_action, self.frame_outcome)
            self.frame_outcome = ActionOutcome()
        elif self.state == "training":
            self.frame_outcome = ActionOutcome()
        elif self.state == "twin":
            if self.previous_twin_action is not None and current_action != self.previous_twin_action:
                self.twin_action_changes += 1
            self.previous_twin_action = current_action
            self.sync_sample_timer += dt
            if self.sync_sample_timer >= SAMPLE_INTERVAL and self.live_prediction is not None:
                self.sync_sample_timer %= SAMPLE_INTERVAL
                correct = self.live_prediction == current_action
                self.prediction_confidences.append(self.live_prediction_confidence)
                self.sync_history.append(correct)
                self.sync_history = self.sync_history[-SYNC_WINDOW_SAMPLES:]
                self.synchronization = sum(self.sync_history) / len(self.sync_history)
                if self.current_arena_elapsed - self.last_prediction_event >= 1.5:
                    event_name = (
                        f"Прогноз верен: {ACTION_LABELS_RU[current_action]}"
                        if correct
                        else f"Ошибка прогноза: {ACTION_LABELS_RU[self.live_prediction]}"
                    )
                    self.replay_recorder.add_event(self.arena_id, self.current_arena_elapsed, event_name)
                    self.last_prediction_event = self.current_arena_elapsed

        if self.state in ("training", "twin"):
            self.replay_recorder.update(
                dt,
                self.arena_id,
                self.current_arena_elapsed,
                self.player.position,
                self.arena.bounds,
                current_action,
            )

        if not self.player.alive:
            self.notice = "СИГНАЛ ПОТЕРЯН — ПЕРЕЗАПУСК АРЕНЫ"
            self.notice_timer = 2.0
            self.respawn_timer = 1.5
            return

        if not any(enemy.alive for enemy in self.enemies):
            if self.state == "twin" and self.round_clear_timer == 0:
                self.slow_motion_timer = 0.55
                self.notice = "КРИТИЧЕСКИЙ СИГНАЛ ДВОЙНИКА"
                self.notice_timer = 1.0
            self.round_clear_timer += dt
            if self.round_clear_timer >= 1.1:
                if self.state == "training":
                    if not self.wave_counted:
                        self.enemies_defeated += len(self.enemies)
                        self.wave_counted = True
                    if self.round_time_left <= 0:
                        if self.objective and self.objective.complete:
                            self._advance_training()
                            return
                        elif self.objective and self.objective.failed:
                            self._setup_arena()
                        else:
                            self.notice = "ЗАВЕРШИТЕ ЗАДАНИЕ АРЕНЫ"
                            self.notice_timer = max(self.notice_timer, 0.4)
                    else:
                        self._spawn_training_wave()
                elif self.state == "twin":
                    self.replay_recorder.save(DATA_DIR / "last_replay.json")
                    self.previous_replay = self.replay_recorder
                    self.state = "result"
                else:
                    if self.custom_mode == "elimination":
                        self._complete_custom_arena("Все противники уничтожены")
                    else:
                        layout = self.editor.layout
                        self.enemies = self._create_custom_enemies()
                        self.round_clear_timer = 0.0
                        self.player.health = min(self.player.max_health, self.player.health + 18)
                        self.notice = "НОВАЯ ВОЛНА"
                        self.notice_timer = 1.0
        else:
            self.round_clear_timer = 0.0
            self.wave_counted = False

    def _create_custom_enemies(self) -> list[Enemy]:
        enemies = [Enemy((x, y), kind) for x, y, kind in self.editor.layout.enemy_spawns]
        if not enemies:
            enemies = [Enemy((self.arena.bounds.right - 140, self.arena.bounds.centery), "assault")]
        for enemy in enemies:
            enemy.position = self.arena.place_circle(enemy.position, enemy.radius)
        return enemies

    def _complete_custom_arena(self, message: str) -> None:
        self.editor.message = message
        self.state = "editor"

    def _update_projectiles(self, dt: float) -> None:
        active: list[Projectile] = []
        for projectile in self.projectiles:
            if not projectile.update(dt, self.arena):
                if projectile.impact_position is not None:
                    self.combat_fx.hit(projectile.impact_position, COLORS["warning"], projectile.damage, strong=True)
                    self.audio.play("hit")
                continue
            hit = False
            if projectile.owner == "player":
                for enemy in self.enemies:
                    if enemy.alive and segment_intersects_circle(
                        projectile.previous_position,
                        projectile.position,
                        enemy.position,
                        enemy.radius + projectile.radius,
                    ):
                        was_alive = enemy.alive
                        applied = enemy.take_damage(projectile.damage, self.player.position)
                        if self.state == "training":
                            self.collector.record_outcome(
                                PlayerAction.RANGED_ATTACK,
                                ActionOutcome(damage_dealt=applied, kill=int(was_alive and not enemy.alive)),
                            )
                            self.calibration_challenge.record("ranged_hit")
                        self.combat_fx.hit(enemy.position, COLORS["player"], applied, strong=applied >= 18)
                        self.audio.play("hit")
                        hit = True
                        break
            elif self.player.alive and segment_intersects_circle(
                projectile.previous_position,
                projectile.position,
                self.player.position,
                self.player.radius + projectile.radius + 8,
            ):
                if self.player.reflect_timer > 0:
                    projectile.owner = "player"
                    projectile.velocity *= -1.18
                    projectile.color = COLORS["player"]
                    projectile.position += projectile.velocity.normalize() * 16
                    self.combat_fx.wave(self.player.position, COLORS["white"])
                else:
                    previous = self.player.health
                    self.player.take_damage(projectile.damage)
                    applied = previous - self.player.health
                    if applied > 0:
                        self.frame_outcome.damage_taken += applied
                        if self.player.block_active:
                            self.frame_outcome.successful_block += 1
                            if self.state == "training":
                                self.calibration_challenge.record("block")
                        self.combat_fx.hit(self.player.position, COLORS["enemy"], applied, strong=applied >= 18)
                        self.audio.play("hit")
                        self.replay_recorder.add_event(self.arena_id, self.current_arena_elapsed, f"Получен урон: {applied:.0f}")
                    hit = True
            if not hit:
                active.append(projectile)
        self.projectiles = active

    def _update_twin_abilities(self, dt: float, twin: Enemy) -> None:
        self.twin_ability_timer -= dt
        if self.twin_ability_timer > 0 or not self.selected_abilities:
            return
        ability = random.choice(self.selected_abilities)
        self.twin_ability_timer = 5.2 if self.twin_phase < 3 else 3.7
        if ability == AbilityType.TRAP:
            self.hazards.append(TrapHazard(twin.position.copy(), "enemy"))
        elif ability == AbilityType.WAVE and twin.position.distance_to(self.player.position) < 190:
            previous = self.player.health
            self.player.take_damage(18)
            self.combat_fx.hit(self.player.position, COLORS["twin"], previous - self.player.health, strong=True)
        elif ability == AbilityType.TELEPORT:
            twin._teleport(self.arena)
        elif ability == AbilityType.SHIELD:
            twin.health = min(twin.max_health, twin.health + 10)
        elif ability == AbilityType.SLOW:
            self.player.energy = max(0.0, self.player.energy - 18)
        elif ability == AbilityType.DECOY:
            self.hazards.append(TrapHazard(twin.position.copy(), "enemy", lifetime=5.0, damage=15))
        elif ability == AbilityType.REFLECT:
            twin.health = min(twin.max_health, twin.health + 6)
        self.notice = f"ДВОЙНИК: {ABILITY_INFO[ability][0]}"
        self.notice_timer = 0.9

    def _update_copier_ability(self, copier: Enemy) -> None:
        if copier.copied_ability is None or copier.copy_use_cooldown > 0:
            return
        try:
            ability = AbilityType(copier.copied_ability)
        except ValueError:
            return
        copier.copy_use_cooldown = 5.0
        if ability == AbilityType.TRAP:
            self.hazards.append(TrapHazard(copier.position.copy(), "enemy"))
        elif ability == AbilityType.WAVE and copier.position.distance_to(self.player.position) < 190:
            self.player.take_damage(16)
            self.combat_fx.wave(copier.position, copier.color)
        elif ability == AbilityType.TELEPORT:
            copier._teleport(self.arena)
        elif ability in (AbilityType.SHIELD, AbilityType.REFLECT):
            copier.health = min(copier.max_health, copier.health + 9)
        elif ability == AbilityType.SLOW:
            self.player.energy = max(0.0, self.player.energy - 15)
        elif ability == AbilityType.DECOY:
            self.hazards.append(TrapHazard(copier.position.copy(), "enemy", lifetime=4.0, damage=12))
        self.notice = f"КОПИРОВЩИК УКРАЛ: {ABILITY_INFO[ability][0]}"
        self.notice_timer = 1.1

    def _advance_training(self) -> None:
        if self.arena_id < 3:
            self.pending_arena_id = self.arena_id + 1
            self.upgrade_choices = (
                [UpgradeType.HEALTH, UpgradeType.DAMAGE, UpgradeType.SPEED]
                if self.pending_arena_id == 2
                else [UpgradeType.ENERGY, UpgradeType.COOLDOWN, UpgradeType.HEAL]
            )
            self.state = "upgrade"
            return
        self.profile = BehaviorProfile.from_samples(self.collector.samples)
        self.data_quality = analyze_data_quality(self.collector.samples)
        self._save_data()
        if self.external_model is not None:
            self.state = "profile"
            return
        self._start_ai_training()

    def _load_metrics_history(self) -> list[dict]:
        path = MODEL_DIR / "metrics_history.json"
        if not path.exists():
            return []
        try:
            return list(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return []

    def _start_ai_training(self) -> None:
        if self.ai_training_thread is not None and self.ai_training_thread.is_alive():
            return
        samples = list(self.collector.samples)
        demo_mode = self.demo_mode
        self.ai_training_error = ""
        self.ai_training_result = None
        self.ai_training_progress = {"stage": "подготовка данных", "epoch": 0, "total": 1}
        self.state = "training_model"

        def progress(stage: str, epoch: int, total: int, point: dict) -> None:
            self.ai_training_progress = {
                "stage": "общая модель" if stage == "general" else "персонализация",
                "epoch": epoch,
                "total": total,
                **point,
            }

        def worker() -> None:
            try:
                if train_gru_pipeline is None:
                    raise RuntimeError("PyTorch недоступен")
                predictor, metrics = train_gru_pipeline(
                    DATA_DIR,
                    samples,
                    MODEL_DIR,
                    demo_mode=demo_mode,
                    progress_callback=progress,
                )
                self.ai_training_result = (predictor, metrics, "")
            except Exception as error:
                neural_model = NeuralActionModel()
                metrics = neural_model.fit(
                    samples,
                    epochs=4 if demo_mode else 6,
                    maximum_samples=350 if demo_mode else 500,
                )
                neural_model.save(MODEL_DIR / "digital_twin_mlp_fallback.json", metrics)
                self.ai_training_result = (neural_model, metrics, str(error))

        self.ai_training_thread = threading.Thread(target=worker, name="digital-twin-training", daemon=True)
        self.ai_training_thread.start()

    def _poll_ai_training(self) -> None:
        if self.ai_training_result is None:
            return
        predictor, metrics, error = self.ai_training_result
        self.ai_training_result = None
        self.predictor = predictor
        self.model_metrics = metrics
        self.ai_training_error = error
        self.neural_model_active = True
        self.metrics_history = self._load_metrics_history()
        self.ai_training_thread = None
        if self.state == "training_model":
            self.state = "profile"

    def _save_data(self) -> None:
        if self.collector.samples:
            self.saved_paths = self.collector.save(DATA_DIR)
            self.replay_recorder.save(DATA_DIR / f"replay_{self.collector.session_id}.json")
            if not self.session_registered:
                self.player_registry.record_session(self.collector.player_id)
                self.session_registered = True

    def _draw(self) -> None:
        if self.state == "menu":
            self._draw_menu()
        elif self.state == "player_select":
            self._draw_player_select()
        elif self.state == "tutorial":
            self._draw_tutorial()
        elif self.state == "tutorial_play":
            self._draw_interactive_tutorial()
        elif self.state == "weapon_select":
            self._draw_weapon_select()
        elif self.state == "upgrade":
            self._draw_upgrade_select()
        elif self.state == "loadout":
            self._draw_loadout()
        elif self.state in ("training", "twin", "custom"):
            self._draw_combat()
        elif self.state == "profile":
            self._draw_profile()
        elif self.state == "training_model":
            self._draw_model_training()
        elif self.state == "model_analysis":
            self._draw_model_analysis()
        elif self.state == "result":
            self._draw_result()
        elif self.state == "replay":
            self._draw_replay()
        elif self.state == "editor":
            self.editor.draw(self.screen, draw_text, self.animated_background.draw, self._mouse_position())
        elif self.state == "lab":
            self._draw_lab()
        elif self.state == "tournament":
            self._draw_tournament()

    def _draw_action_button(self, rect: pygame.Rect, label: str, *, enabled: bool = True) -> None:
        hovered = enabled and rect.collidepoint(self._mouse_position())
        if enabled:
            fill = (25, 71, 75) if hovered else (20, 48, 61)
            border = COLORS["player"]
            text_color = COLORS["white"]
        else:
            fill = (17, 27, 42)
            border = (51, 67, 91)
            text_color = COLORS["muted"]
        animated_rect = rect.inflate(6, 4) if hovered else rect
        logical_draw.rect(self.screen, fill, animated_rect, border_radius=11)
        logical_draw.rect(self.screen, border, animated_rect, 2, border_radius=11)
        draw_text(self.screen, label, animated_rect.center, 18, text_color, bold=True, anchor="center")

    def _draw_menu(self) -> None:
        self.animated_background.draw(self.screen, 1.0)
        float_y = 0.0
        pulse = 0.65
        for index in range(3):
            radius = int(92 + index * 31 + pulse * 9)
            ring_color = (12 + index * 5, 45 + index * 7, 57 + index * 9)
            logical_draw.circle(self.screen, ring_color, (WIDTH // 2, 142 + int(float_y)), radius, 1)
        draw_text(self.screen, "ПРОТОКОЛ:", (WIDTH / 2, 78 + float_y), 25, COLORS["muted"], bold=True, anchor="midtop")
        glow = (20, int(82 + pulse * 35), int(91 + pulse * 40))
        draw_text(self.screen, "ДВОЙНИК", (WIDTH / 2 + 2, 108 + float_y), 68, glow, bold=True, anchor="midtop")
        draw_text(self.screen, "ДВОЙНИК", (WIDTH / 2, 105 + float_y), 68, COLORS["player"], bold=True, anchor="midtop")
        draw_text(self.screen, "Игра изучает твои привычки. Финальный противник — ты сам.", (WIDTH / 2, 198), 21, anchor="midtop")
        profile_rect = self._profile_badge_rect()
        profile_hovered = profile_rect.collidepoint(self._mouse_position())
        logical_draw.rect(self.screen, (17, 48, 59) if profile_hovered else (10, 25, 38), profile_rect, border_radius=9)
        logical_draw.rect(self.screen, COLORS["player"] if profile_hovered else (40, 91, 103), profile_rect, 1, border_radius=9)
        visible_profile_name = self.current_player.name
        if len(visible_profile_name) > 18:
            visible_profile_name = visible_profile_name[:17] + "…"
        draw_text(
            self.screen,
            f"ПРОФИЛЬ: {visible_profile_name}   •   ИЗМЕНИТЬ",
            profile_rect.center,
            14,
            COLORS["player"],
            bold=True,
            anchor="center",
        )
        labels = ["НАЧАТЬ ИГРУ", "РЕДАКТОР", "ОБУЧЕНИЕ МЕХАНИКАМ"]
        mouse_position = self._mouse_position()
        for index, (label, rect) in enumerate(zip(labels, self._menu_button_rects())):
            selected = index == self.menu_selected or rect.collidepoint(mouse_position)
            target = 1.0 if selected else 0.0
            self.menu_focus[index] += (target - self.menu_focus[index]) * (1.0 - math.exp(-11.0 * self.frame_dt))
            focus = self.menu_focus[index]
            animated_rect = rect.copy()
            animated_rect.inflate_ip(int(focus * 10), int(focus * 4))
            fill = (
                int(18 + focus * 7),
                int(29 + focus * 28),
                int(47 + focus * 22),
            )
            border = tuple(int((58, 78, 106)[channel] + (COLORS["player"][channel] - (58, 78, 106)[channel]) * focus) for channel in range(3))
            logical_draw.rect(self.screen, fill, animated_rect, border_radius=12)
            logical_draw.rect(self.screen, border, animated_rect, 2, border_radius=12)
            if focus > 0.2:
                light_width = int((animated_rect.width - 45) * focus)
                logical_draw.line(self.screen, border, (animated_rect.centerx - light_width // 2, animated_rect.bottom - 5), (animated_rect.centerx + light_width // 2, animated_rect.bottom - 5), 2)
            draw_text(self.screen, label, animated_rect.center, 21, COLORS["white"] if focus > 0.45 else COLORS["text"], bold=True, anchor="center")
        draw_text(self.screen, "Мышь или ↑ ↓ и ENTER", (WIDTH / 2, 593), 16, COLORS["muted"], anchor="midtop")
        demo_color = COLORS["health"] if self.demo_mode else COLORS["muted"]
        demo_text = "ВКЛЮЧЁН" if self.demo_mode else "выключен"
        draw_text(self.screen, f"F9 — конкурсный демо-режим: {demo_text}", (WIDTH / 2, 634), 14, demo_color, bold=self.demo_mode, anchor="midtop")
        display_mode_hint = "оконный режим" if self.fullscreen else "полноэкранный режим"
        draw_text(self.screen, f"F11 — {display_mode_hint}   •   ESC — выход", (WIDTH / 2, 674), 14, COLORS["muted"], anchor="midtop")

    def _draw_player_select(self) -> None:
        self.animated_background.draw(self.screen, 0.72)
        draw_text(self.screen, "ПРОФИЛЬ ИГРОКА", (WIDTH / 2, 42), 39, COLORS["player"], bold=True, anchor="midtop")
        draw_text(
            self.screen,
            "Данные разных людей больше не смешиваются. Выберите профиль или напечатайте новое имя.",
            (WIDTH / 2, 96),
            17,
            COLORS["muted"],
            anchor="midtop",
        )
        mouse = self._mouse_position()
        profiles = self._visible_player_profiles()
        rename_rect = self._player_rename_rect()
        delete_rect = self._player_delete_rect()
        for rect, label, danger in (
            (rename_rect, "ПЕРЕИМЕНОВАТЬ", False),
            (delete_rect, "ПОДТВЕРДИТЬ" if self.player_management_mode == "delete_confirm" else "УДАЛИТЬ", True),
        ):
            hovered = rect.collidepoint(mouse)
            if danger:
                fill = (83, 35, 48) if hovered else (44, 27, 41)
                border = COLORS["enemy"] if hovered or self.player_management_mode == "delete_confirm" else (95, 57, 72)
            else:
                fill = (22, 58, 65) if hovered else (18, 34, 50)
                border = COLORS["player"] if hovered else (61, 82, 108)
            logical_draw.rect(self.screen, fill, rect, border_radius=8)
            logical_draw.rect(self.screen, border, rect, 1, border_radius=8)
            draw_text(self.screen, label, rect.center, 12, COLORS["white"], bold=True, anchor="center")
        for index, (profile, rect) in enumerate(zip(profiles, self._player_card_rects())):
            selected = index == self.player_select_index and not self.player_name_input
            hovered = rect.collidepoint(mouse)
            fill = (22, 61, 67) if selected else ((20, 37, 54) if hovered else COLORS["panel"])
            border = COLORS["player"] if selected else ((76, 103, 132) if hovered else (52, 72, 101))
            logical_draw.rect(self.screen, fill, rect, border_radius=11)
            logical_draw.rect(self.screen, border, rect, 2, border_radius=11)
            visible_name = profile.name if len(profile.name) <= 32 else profile.name[:31] + "…"
            draw_text(self.screen, visible_name, (rect.left + 24, rect.top + 15), 21, COLORS["white"], bold=True)
            draw_text(
                self.screen,
                f"Сессий: {profile.session_count}   •   ID: {profile.player_id}",
                (rect.right - 22, rect.top + 19),
                14,
                COLORS["muted"],
                anchor="topright",
            )
        input_rect = self._player_name_input_rect()
        if self.player_management_mode == "rename":
            field_label = "ПЕРЕИМЕНОВАНИЕ — введите новое имя • Ctrl+V • до 60 символов"
        else:
            field_label = "НОВЫЙ ПРОФИЛЬ — введите имя • Ctrl+V • до 60 символов"
        draw_text(
            self.screen,
            field_label,
            (input_rect.left, input_rect.top - 25),
            14,
            COLORS["player"] if self.player_name_focused else COLORS["muted"],
            bold=self.player_name_focused,
        )
        logical_draw.rect(self.screen, (10, 18, 31), input_rect, border_radius=9)
        input_hovered = input_rect.collidepoint(mouse)
        input_border = COLORS["player"] if self.player_name_focused or input_hovered else (62, 82, 109)
        logical_draw.rect(self.screen, input_border, input_rect, 2, border_radius=9)
        input_text = self.player_name_input or "Введите имя нового игрока…"
        text_x = input_rect.left + 17
        if self.player_name_input:
            physical_font_size = max(1, round(17 * self.render_scale))
            text_width = get_font(physical_font_size).size(input_text)[0] / self.render_scale
            text_x = min(text_x, input_rect.right - 17 - text_width)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(scale_rect(self.screen, input_rect.inflate(-14, -4)))
        text_rect = draw_text(
            self.screen,
            input_text,
            (text_x, input_rect.top + 10),
            17,
            COLORS["white"] if self.player_name_input else COLORS["muted"],
        )
        if self.player_name_focused and int(self.ui_time * 2) % 2 == 0:
            cursor_x = text_rect.right + 2 if self.player_name_input else input_rect.left + 17
            logical_draw.line(self.screen, COLORS["player"], (cursor_x, input_rect.top + 9), (cursor_x, input_rect.bottom - 9), 2)
        self.screen.set_clip(previous_clip)
        continue_rect = self._player_continue_rect()
        logical_draw.rect(self.screen, (25, 65, 68), continue_rect, border_radius=11)
        logical_draw.rect(self.screen, COLORS["player"], continue_rect, 2, border_radius=11)
        continue_label = "СОХРАНИТЬ ИМЯ" if self.player_management_mode == "rename" else "ПРОДОЛЖИТЬ"
        draw_text(self.screen, continue_label, continue_rect.center, 19, COLORS["white"], bold=True, anchor="center")
        if self.player_profile_notice:
            notice_color = COLORS["enemy"] if self.player_management_mode == "delete_confirm" else COLORS["warning"]
            draw_text(self.screen, self.player_profile_notice, (WIDTH / 2, 690), 13, notice_color, anchor="midbottom")
        draw_text(self.screen, "ESC — главное меню", (WIDTH / 2, 716), 13, COLORS["muted"], anchor="midbottom")

    def _draw_tutorial(self) -> None:
        pages = [
            (
                "ДВИЖЕНИЕ И ПРИЦЕЛИВАНИЕ",
                "Управляй персонажем клавишами WASD. Персонаж смотрит в сторону курсора.",
                [("W A S D", "Движение по арене"), ("МЫШЬ", "Направление атак"), ("ЦЕЛЬ", "Не стой на месте — меняй траекторию")],
            ),
            (
                "АТАКА И ЗАЩИТА",
                "Выбирай дистанцию боя и следи за синей шкалой энергии.",
                [("ЛКМ", "Дальний выстрел"), ("ПКМ", "Ближняя атака"), ("SHIFT", "Блок с расходом энергии"), ("Q", "Использовать заряд лечения")],
            ),
            (
                "РЫВОК И ИДЕАЛЬНОЕ УКЛОНЕНИЕ",
                "Нажми SPACE перед самым попаданием: время замедлится, а энергия восстановится.",
                [("SPACE", "Рывок по направлению движения"), ("МOMЕНТ", "Уклоняйся непосредственно перед ударом"), ("НАГРАДА", "+32 энергии за идеальное уклонение")],
            ),
            (
                "СПОСОБНОСТИ И ОБМАН",
                "Перед игрой выбери три способности. Они назначаются на клавиши 1, 2 и 3.",
                [("1 2 3", "Использовать выбранные способности"), ("F", "Ложная атака — сбивает профиль"), ("ТАКТИКА", "Комбинируй способности с рывком и блоком")],
            ),
            (
                "КАЛИБРОВКА И ДВОЙНИК",
                "Пройди три задания. Игра соберёт стиль поведения и создаст финального двойника.",
                [("АРЕНА 1", "Удерживай отмеченную зону"), ("АРЕНА 2", "Собирай энергетические ядра"), ("АРЕНА 3", "Защищай системное ядро"), ("ФИНАЛ", "Меняй привычки, чтобы победить себя")],
            ),
        ]
        title, description, cards = pages[self.tutorial_page]
        self.animated_background.draw(self.screen, 0.72)
        eased = 1.0 - (1.0 - self.tutorial_transition) ** 3
        content_offset = int((1.0 - eased) * 105)
        title_float = 0.0
        draw_text(self.screen, "ОБУЧЕНИЕ", (WIDTH / 2, 35 + title_float), 20, COLORS["player"], bold=True, anchor="midtop")
        draw_text(self.screen, title, (WIDTH / 2 + content_offset, 72 + title_float), 34, COLORS["white"], bold=True, anchor="midtop")
        draw_text(self.screen, description, (WIDTH / 2 + content_offset, 125), 18, COLORS["muted"], anchor="midtop")
        panel = pygame.Rect(135 + content_offset, 185, WIDTH - 270, 390)
        draw_panel(self.screen, panel)
        card_height = 64
        start_y = panel.top + 42
        for index, (key, explanation) in enumerate(cards):
            card_delay = min(1.0, eased * 1.2 - index * 0.04)
            card_shift = int((1.0 - max(0.0, card_delay)) * 40)
            rect = pygame.Rect(panel.left + 55 + card_shift, start_y + index * 78, panel.width - 110, card_height)
            logical_draw.rect(self.screen, (19, 33, 52), rect, border_radius=10)
            logical_draw.rect(self.screen, (53, 76, 105), rect, 1, border_radius=10)
            key_rect = pygame.Rect(rect.left + 14, rect.top + 11, 150, 42)
            logical_draw.rect(self.screen, (24, 62, 68), key_rect, border_radius=8)
            draw_text(self.screen, key, key_rect.center, 16, COLORS["player"], bold=True, anchor="center")
            draw_text(self.screen, explanation, (rect.left + 185, rect.centery), 17, COLORS["text"], anchor="midleft")
        for index in range(len(pages)):
            color = COLORS["player"] if index == self.tutorial_page else (53, 72, 98)
            logical_draw.circle(self.screen, color, (WIDTH // 2 - 48 + index * 24, 608), 6)
        back, forward = self._tutorial_navigation_rects()
        for rect, label in ((back, "← НАЗАД"), (forward, "НАЧАТЬ ПРАКТИКУ" if self.tutorial_page == 4 else "ДАЛЕЕ →")):
            hovered = rect.collidepoint(self._mouse_position())
            logical_draw.rect(self.screen, (24, 55, 67) if hovered else COLORS["panel"], rect, border_radius=9)
            logical_draw.rect(self.screen, COLORS["player"] if hovered else (61, 79, 108), rect, 2, border_radius=9)
            draw_text(self.screen, label, rect.center, 17, COLORS["white"], bold=True, anchor="center")
        draw_text(self.screen, f"{self.tutorial_page + 1}/5", (WIDTH / 2, 653), 16, COLORS["muted"], anchor="midtop")

    def _draw_interactive_tutorial(self) -> None:
        world = self.world_surface
        self.arena.draw(world)
        target = self.enemies[0]
        if self.tutorial_step == 3:
            logical_draw.line(world, COLORS["enemy"], target.position, self.player.position, 2)
            logical_draw.circle(world, COLORS["warning"], self.player.position, self.player.radius + 18, 2)
        for projectile in self.projectiles:
            projectile.draw(world)
        for effect in self.effects:
            effect.draw(world)
        target.draw(world, self.arena.bounds)
        self.player.draw(world)
        self.combat_fx.draw(world, pygame.Vector2())
        logical_blit(self.screen, world, (0, 0))
        logical_draw.rect(self.screen, COLORS["panel"], (0, 0, WIDTH, 82))
        draw_text(self.screen, "ПРАКТИЧЕСКОЕ ОБУЧЕНИЕ", (28, 18), 24, COLORS["player"], bold=True)
        draw_text(self.screen, "ESC — выйти", (WIDTH - 28, 25), 15, COLORS["muted"], anchor="topright")
        if self.tutorial_step >= len(INTERACTIVE_TUTORIAL_STEPS):
            panel = pygame.Rect(WIDTH // 2 - 330, HEIGHT // 2 - 115, 660, 230)
            draw_panel(self.screen, panel)
            draw_text(self.screen, "ОБУЧЕНИЕ ЗАВЕРШЕНО", (WIDTH / 2, panel.top + 42), 31, COLORS["health"], bold=True, anchor="midtop")
            draw_text(self.screen, "Все основные механики успешно выполнены.", (WIDTH / 2, panel.top + 98), 18, COLORS["text"], anchor="midtop")
            draw_text(self.screen, "ENTER — ВЕРНУТЬСЯ В МЕНЮ", (WIDTH / 2, panel.top + 158), 19, COLORS["warning"], bold=True, anchor="midtop")
            return
        title, description, key = INTERACTIVE_TUTORIAL_STEPS[self.tutorial_step]
        panel = pygame.Rect(WIDTH // 2 - 360, HEIGHT - 132, 720, 92)
        draw_panel(self.screen, panel)
        draw_text(self.screen, f"{self.tutorial_step + 1}/{len(INTERACTIVE_TUTORIAL_STEPS)}  {title}", (panel.left + 22, panel.top + 16), 20, COLORS["warning"], bold=True)
        draw_text(self.screen, description, (panel.left + 22, panel.top + 49), 15, COLORS["text"])
        key_rect = pygame.Rect(panel.right - 125, panel.top + 18, 98, 55)
        logical_draw.rect(self.screen, (25, 60, 67), key_rect, border_radius=9)
        logical_draw.rect(self.screen, COLORS["player"], key_rect, 2, border_radius=9)
        draw_text(self.screen, key, key_rect.center, 17, COLORS["player"], bold=True, anchor="center")
        if self.tutorial_step == 0:
            progress = self.tutorial_progress / 1.5
        elif self.tutorial_step == 1:
            progress = self.tutorial_hits / 3
        elif self.tutorial_step == 3:
            progress = self.tutorial_progress
        else:
            progress = 0.0
        if progress > 0:
            bar = pygame.Rect(panel.left + 22, panel.bottom - 9, panel.width - 44, 4)
            logical_draw.rect(self.screen, (35, 54, 75), bar)
            logical_draw.rect(self.screen, COLORS["player"], (bar.x, bar.y, int(bar.width * min(1.0, progress)), bar.height))

    def _draw_loadout(self) -> None:
        self.animated_background.draw(self.screen, 0.68)
        intro_eased = 1.0 - (1.0 - self.loadout_intro) ** 3
        title_offset = int((1.0 - intro_eased) * -34)
        title_float = 0.0
        ring_pulse = 0.65
        for index in range(3):
            radius = 55 + index * 22 + int(ring_pulse * 5)
            logical_draw.circle(self.screen, (13 + index * 4, 42 + index * 7, 55 + index * 8), (WIDTH // 2, 62 + title_offset), radius, 1)
        draw_text(self.screen, "ВЫБЕРИТЕ ТРИ СПОСОБНОСТИ", (WIDTH / 2, 42 + title_offset + title_float), 37, COLORS["player"], bold=True, anchor="midtop")
        draw_text(self.screen, "Нажимайте 1–7 или выбирайте мышью. Двойник получит тот же набор.", (WIDTH / 2, 96 + title_offset), 18, COLORS["muted"], anchor="midtop")
        abilities = list(AbilityType)
        mouse_position = self._mouse_position()
        for index, (ability, base_rect) in enumerate(zip(abilities, self._loadout_card_rects())):
            progress = max(0.0, min(1.0, (self.loadout_intro - index * 0.045) / 0.73))
            card_eased = 1.0 - (1.0 - progress) ** 3
            direction = -1 if index % 2 == 0 else 1
            rect = base_rect.move(int(direction * (1.0 - card_eased) * 135), 0)
            selected = ability in self.loadout_selection
            hovered = base_rect.collidepoint(mouse_position)
            target_focus = 1.0 if selected else (0.48 if hovered else 0.0)
            self.loadout_focus[index] += (target_focus - self.loadout_focus[index]) * (1.0 - math.exp(-12.0 * self.frame_dt))
            focus = self.loadout_focus[index]
            rect.inflate_ip(int(focus * 7), int(focus * 4))
            fill = (int(17 + focus * 8), int(25 + focus * 31), int(43 + focus * 23))
            neutral_border = (61, 79, 108)
            border = tuple(int(neutral_border[channel] + (COLORS["player"][channel] - neutral_border[channel]) * focus) for channel in range(3))
            logical_draw.rect(self.screen, fill, rect, border_radius=12)
            logical_draw.rect(self.screen, border, rect, 2, border_radius=12)
            if selected:
                track_start = (rect.left + 25, rect.bottom - 5)
                track_end = (rect.right - 25, rect.bottom - 5)
                logical_draw.line(self.screen, (34, 79, 82), track_start, track_end, 2)
                filled_width = int((rect.width - 50) * self.loadout_bar_progress[index])
                if filled_width > 0:
                    logical_draw.line(self.screen, border, track_start, (track_start[0] + filled_width, track_start[1]), 2)
            name, description, cooldown = ABILITY_INFO[ability]
            draw_text(self.screen, str(index + 1), (rect.left + 25, rect.centery), 25, COLORS["warning"], bold=True, anchor="center")
            draw_text(self.screen, name, (rect.left + 58, rect.top + 15), 20, COLORS["white"], bold=True)
            draw_text(self.screen, description, (rect.left + 58, rect.top + 47), 15, COLORS["muted"])
            draw_text(self.screen, f"{cooldown:.0f}с", (rect.right - 20, rect.top + 17), 15, COLORS["muted"], anchor="topright")
        status = f"Выбрано: {len(self.loadout_selection)}/3"
        draw_text(self.screen, status, (WIDTH / 2, 615), 21, COLORS["health"] if len(self.loadout_selection) == 3 else COLORS["warning"], bold=True, anchor="midtop")
        ready = len(self.loadout_selection) == 3
        self._draw_action_button(
            self._loadout_continue_rect(),
            "НАЧАТЬ" if ready else "НУЖНО ВЫБРАТЬ РОВНО ТРИ",
            enabled=ready,
        )

    def _draw_weapon_select(self) -> None:
        self.animated_background.draw(self.screen, 0.68)
        draw_text(self.screen, "ВЫБЕРИТЕ ОРУЖИЕ", (WIDTH / 2, 54), 39, COLORS["player"], bold=True, anchor="midtop")
        draw_text(self.screen, "Оружие влияет на дистанцию, темп и поведенческий профиль.", (WIDTH / 2, 108), 18, COLORS["muted"], anchor="midtop")
        mouse_position = self._mouse_position()
        for index, (weapon, base_rect) in enumerate(zip(WeaponType, self._weapon_card_rects())):
            progress = max(0.0, min(1.0, (self.weapon_intro - index * 0.07) / 0.72))
            eased = 1.0 - (1.0 - progress) ** 3
            direction = -1 if index % 2 == 0 else 1
            rect = base_rect.move(int(direction * (1.0 - eased) * 150), 0)
            selected = weapon == self.selected_weapon
            hovered = base_rect.collidepoint(mouse_position)
            fill = (23, 55, 64) if selected else ((20, 35, 53) if hovered else COLORS["panel"])
            border = COLORS["player"] if selected else (75, 101, 135) if hovered else (56, 75, 103)
            logical_draw.rect(self.screen, fill, rect, border_radius=13)
            logical_draw.rect(self.screen, border, rect, 2, border_radius=13)
            spec = WEAPON_SPECS[weapon]
            draw_text(self.screen, str(index + 1), (rect.left + 28, rect.centery), 28, COLORS["warning"], bold=True, anchor="center")
            draw_text(self.screen, spec.name, (rect.left + 60, rect.top + 20), 22, COLORS["white"], bold=True)
            draw_text(self.screen, spec.description, (rect.left + 60, rect.top + 56), 15, COLORS["muted"])
            stats = f"Урон {spec.ranged_damage:.0f}   •   Ближний {spec.melee_damage:.0f}   •   Энергия {spec.energy_cost:.0f}"
            draw_text(self.screen, stats, (rect.left + 60, rect.top + 87), 14, COLORS["player"] if selected else COLORS["muted"])
        draw_text(self.screen, f"Выбрано: {WEAPON_SPECS[self.selected_weapon].name}", (WIDTH / 2, 565), 20, COLORS["health"], bold=True, anchor="midtop")
        self._draw_action_button(self._weapon_continue_rect(), "ВЫБРАТЬ СПОСОБНОСТИ")
        draw_text(self.screen, "ESC — главное меню", (WIDTH / 2, 672), 14, COLORS["muted"], anchor="midtop")

    def _draw_upgrade_select(self) -> None:
        self.animated_background.draw(self.screen, 0.62)
        draw_text(self.screen, "УСИЛЕНИЕ ПРОТОКОЛА", (WIDTH / 2, 58), 38, COLORS["player"], bold=True, anchor="midtop")
        draw_text(self.screen, f"Калибровка {self.pending_arena_id - 1} завершена. Выберите одно постоянное улучшение.", (WIDTH / 2, 112), 18, COLORS["muted"], anchor="midtop")
        mouse_position = self._mouse_position()
        for index, (upgrade, rect) in enumerate(zip(self.upgrade_choices, self._upgrade_card_rects())):
            hovered = rect.collidepoint(mouse_position)
            fill = (24, 52, 61) if hovered else COLORS["panel"]
            border = COLORS["player"] if hovered else (60, 81, 111)
            logical_draw.rect(self.screen, fill, rect, border_radius=14)
            logical_draw.rect(self.screen, border, rect, 2, border_radius=14)
            name, description = UPGRADE_INFO[upgrade]
            draw_text(self.screen, str(index + 1), (rect.centerx, rect.top + 24), 27, COLORS["warning"], bold=True, anchor="midtop")
            draw_text(self.screen, name, (rect.centerx, rect.top + 77), 21, COLORS["white"], bold=True, anchor="midtop")
            draw_text(self.screen, description, (rect.centerx, rect.top + 127), 15, COLORS["muted"], anchor="midtop")
            logical_draw.line(self.screen, border, (rect.left + 35, rect.bottom - 35), (rect.right - 35, rect.bottom - 35), 2)
        chosen_names = [UPGRADE_INFO[UpgradeType(value)][0] for value in self.session_upgrades]
        history = "Уже установлено: " + (" • ".join(chosen_names) if chosen_names else "нет")
        draw_text(self.screen, history, (WIDTH / 2, 545), 16, COLORS["health"], anchor="midtop")
        draw_text(self.screen, "Нажмите 1–3 или выберите мышью", (WIDTH / 2, 615), 17, COLORS["text"], anchor="midtop")

    def _draw_background_nodes(self) -> None:
        random.seed(11)
        points = [(random.randint(40, WIDTH - 40), random.randint(50, HEIGHT - 50)) for _ in range(32)]
        for index, point in enumerate(points):
            for other in points[index + 1 :]:
                if pygame.Vector2(point).distance_to(other) < 155:
                    logical_draw.line(self.screen, (19, 48, 65), point, other, 1)
            logical_draw.circle(self.screen, (31, 91, 102), point, 3)

    def _draw_combat(self) -> None:
        world = self.world_surface
        self.arena.draw(world)
        if self.state == "custom" and self.custom_mode == "hold":
            objective = self.editor.layout.objective_position
            inside = self.player.position.distance_to(pygame.Vector2(objective)) <= 90
            zone_color = COLORS["health"] if inside else COLORS["warning"]
            logical_draw.circle(world, tuple(max(7, channel // 8) for channel in zone_color), objective, 90)
            logical_draw.circle(world, zone_color, objective, 90, 3)
        if self.objective:
            self.objective.draw(world)
        if self.state == "training" and self.previous_replay:
            ghost = self.previous_replay.position_at(self.arena_id, self.current_arena_elapsed, self.arena.bounds)
            if ghost is not None:
                layer = self.alpha_surface
                layer.fill((0, 0, 0, 0))
                logical_draw.circle(layer, (*COLORS["twin"], 70), ghost, 18)
                logical_draw.circle(layer, (*COLORS["white"], 90), ghost, 26, 2)
                logical_blit(world, layer, (0, 0))
        for hazard in self.hazards:
            hazard.draw(world)
        for decoy in self.decoys:
            decoy.draw(world)
        for projectile in self.projectiles:
            projectile.draw(world)
        for effect in self.effects:
            effect.draw(world)
        for enemy in self.enemies:
            if enemy.alive:
                enemy.draw(world, self.arena.bounds)
        self.player.draw(world)
        self.combat_fx.draw(world, pygame.Vector2())
        offset = self.combat_fx.offset()
        if offset.length_squared():
            self.screen.fill(COLORS["background"])
        logical_blit(self.screen, world, offset)
        self._draw_hud()
        if self.round_clear_timer > 0:
            message = "ВОЛНА НЕЙТРАЛИЗОВАНА" if self.state == "training" else "СИГНАЛ НЕЙТРАЛИЗОВАН"
            draw_text(self.screen, message, (WIDTH / 2, HEIGHT / 2 - 20), 34, COLORS["health"], bold=True, anchor="center")
        if self.combo_notice_timer > 0:
            draw_text(self.screen, self.combo_notice, (WIDTH / 2, HEIGHT - 56), 20, COLORS["warning"], bold=True, anchor="midbottom")
        if self.notice_timer > 0:
            rect = pygame.Rect(WIDTH // 2 - 320, 94, 640, 44)
            draw_panel(self.screen, rect, 225)
            draw_text(self.screen, self.notice, rect.center, 18, COLORS["warning"], bold=True, anchor="center")
        if self.state == "twin" and self.twin_intro_timer > 0:
            self._draw_twin_intro()

    def _draw_hud(self) -> None:
        logical_draw.rect(self.screen, COLORS["panel"], (0, 0, WIDTH, 78))
        draw_text(self.screen, "ДВОЙНИК", (32, 18), 25, COLORS["player"], bold=True)
        fps = self.clock.get_fps()
        fps_color = COLORS["health"] if fps >= 55 else (COLORS["warning"] if fps >= 35 else COLORS["enemy"])
        draw_text(self.screen, f"FPS: {fps:04.1f}", (32, 51), 13, fps_color, bold=True)
        draw_bar(self.screen, pygame.Rect(190, 18, 230, 14), self.player.health, self.player.max_health, COLORS["health"], "ЗДОРОВЬЕ")
        draw_bar(self.screen, pygame.Rect(190, 51, 230, 10), self.player.energy, self.player.max_energy, COLORS["energy"], "ЭНЕРГИЯ")
        if self.state == "training":
            prefix = "ДЕМО • " if self.demo_mode else ""
            mode = f"{prefix}КАЛИБРОВКА {self.arena_id}/3"
        elif self.state == "twin":
            phase_names = {1: "КОПИРОВАНИЕ", 2: "ПРЕДСКАЗАНИЕ", 3: "ИСКАЖЕНИЕ"}
            mode = f"ДВОЙНИК • {phase_names[self.twin_phase]}"
        else:
            mode = f"РЕДАКТОР • {GAME_MODE_NAMES.get(self.custom_mode, 'Уничтожение').upper()}"
        draw_text(self.screen, mode, (WIDTH / 2, 8), 18, COLORS["warning"] if self.state == "twin" else COLORS["text"], bold=True, anchor="midtop")
        if self.state == "training":
            timer_text = (
                "ЗАЧИСТИТЕ ОСТАВШИХСЯ"
                if self.round_time_left <= 0
                else f"{math.ceil(self.round_time_left):02d} сек.  •  волна {self.wave_number}"
            )
            draw_text(self.screen, timer_text, (WIDTH / 2, 38), 17, COLORS["warning"], bold=True, anchor="midtop")
            draw_text(self.screen, f"Собрано примеров: {len(self.collector.samples)}", (WIDTH - 32, 18), 17, COLORS["muted"], anchor="topright")
            draw_text(self.screen, "5 записей/сек", (WIDTH - 32, 45), 15, COLORS["muted"], anchor="topright")
        elif self.state == "twin":
            sync_rect = pygame.Rect(WIDTH // 2 - 120, 48, 240, 9)
            draw_bar(self.screen, sync_rect, self.synchronization, 1.0, COLORS["twin"])
            draw_text(self.screen, f"Синхронизация {self.synchronization * 100:.0f}%", (WIDTH / 2, 62), 13, COLORS["muted"], anchor="midbottom")
            draw_text(self.screen, f"Стиль: {self.profile.style_name}", (WIDTH - 32, 18), 17, COLORS["twin"], anchor="topright")
            prediction = ACTION_LABELS_RU[self.live_prediction] if self.live_prediction is not None else "анализ"
            model_label = "нейросеть" if self.neural_model_active else "правила"
            draw_text(self.screen, f"Прогноз: {prediction} {self.live_prediction_confidence * 100:.0f}% • {model_label}", (WIDTH - 32, 45), 14, COLORS["muted"], anchor="topright")
        else:
            draw_text(self.screen, f"Врагов: {sum(enemy.alive for enemy in self.enemies)}", (WIDTH - 32, 28), 17, COLORS["muted"], anchor="topright")
            if self.custom_mode == "survival":
                custom_status = f"Осталось: {math.ceil(self.custom_time_left)} сек."
            elif self.custom_mode == "hold":
                custom_status = f"Удержание: {self.custom_hold_progress:.1f}/15 сек."
            else:
                custom_status = "Уничтожьте всех противников"
            draw_text(self.screen, custom_status, (WIDTH / 2, 39), 15, COLORS["warning"], bold=True, anchor="midtop")
        weapon_name = WEAPON_SPECS[self.player.weapon].name
        draw_text(self.screen, f"{weapon_name}   •   Лечение [Q]: {self.player.heal_charges}   •   Обман [F]", (self.arena.bounds.left, HEIGHT - 32), 14, COLORS["muted"])
        if self.objective:
            draw_text(self.screen, self.objective.status(), (WIDTH / 2, HEIGHT - 32), 15, COLORS["warning"], bold=True, anchor="midtop")
        if self.state == "training" and self.calibration_challenge.goals:
            challenge_text = "   •   ".join(
                f"{label}: {value}/{target}" for label, value, target in self.calibration_challenge.lines()
            )
            challenge_color = COLORS["health"] if self.calibration_challenge.complete else COLORS["player"]
            draw_text(
                self.screen,
                f"AI-ИСПЫТАНИЕ   {challenge_text}",
                (WIDTH / 2, HEIGHT - 55),
                13,
                challenge_color,
                bold=True,
                anchor="midtop",
            )
        ability_x = self.arena.bounds.right
        for reverse_index, ability in enumerate(reversed(self.ability_system.selected)):
            slot = len(self.ability_system.selected) - reverse_index
            cooldown = self.ability_system.cooldowns[ability]
            text = f"[{slot}] {ABILITY_INFO[ability][0]}"
            if cooldown > 0:
                text += f" {cooldown:.1f}с"
            color = COLORS["enemy"] if self.player.ability_lock_timer > 0 else (COLORS["muted"] if cooldown > 0 else COLORS["player"])
            rect = draw_text(self.screen, text, (ability_x, HEIGHT - 32), 14, color, anchor="topright")
            ability_x = rect.left - 18

    def _draw_twin_intro(self) -> None:
        twin = next((enemy for enemy in self.enemies if enemy.kind == "twin"), None)
        if twin is None:
            return
        progress = 1.0 - self.twin_intro_timer / TWIN_INTRO_SECONDS
        overlay = self.alpha_surface
        overlay.fill((0, 0, 0, 0))
        overlay.fill((4, 2, 14, int(150 * (1.0 - progress * 0.45))))
        for index in range(4):
            radius = int(28 + ((progress * 220 + index * 52) % 225))
            alpha = max(20, 170 - radius // 2)
            logical_draw.circle(overlay, (*COLORS["twin"], alpha), twin.position, radius, 2)
        logical_blit(self.screen, overlay, (0, 0))
        jitter = int(math.sin(progress * 80) * 8)
        for index in range(9):
            y = int(twin.position.y - 120 + index * 28 + math.sin(progress * 35 + index) * 8)
            width = 110 + (index % 3) * 36
            logical_draw.line(self.screen, COLORS["twin"], (twin.position.x - width + jitter, y), (twin.position.x + width, y), 2)
        draw_text(self.screen, "СОЗДАНИЕ ЦИФРОВОГО ПРОФИЛЯ", (WIDTH / 2, 175), 23, COLORS["muted"], bold=True, anchor="midtop")
        draw_text(self.screen, f"{int(progress * 100):02d}%", (WIDTH / 2, 212), 58, COLORS["twin"], bold=True, anchor="midtop")

    def _draw_heatmap(self, rect: pygame.Rect) -> None:
        logical_draw.rect(self.screen, (10, 17, 30), rect, border_radius=8)
        maximum = max(1, self.heatmap.maximum)
        cell_width = rect.width / self.heatmap.columns
        cell_height = rect.height / self.heatmap.rows
        for row_index, row in enumerate(self.heatmap.cells):
            for column_index, count in enumerate(row):
                if count <= 0:
                    continue
                intensity = math.sqrt(count / maximum)
                color = (
                    int(23 + (COLORS["player"][0] - 23) * intensity),
                    int(35 + (COLORS["player"][1] - 35) * intensity),
                    int(57 + (COLORS["twin"][2] - 57) * intensity),
                )
                cell = pygame.Rect(
                    rect.left + column_index * cell_width,
                    rect.top + row_index * cell_height,
                    math.ceil(cell_width) + 1,
                    math.ceil(cell_height) + 1,
                )
                logical_draw.rect(self.screen, color, cell)
        logical_draw.rect(self.screen, (72, 91, 118), rect, 1, border_radius=8)

    def _draw_confusion_matrix(self, rect: pygame.Rect) -> None:
        matrix = getattr(self.model_metrics, "confusion_matrix", ())
        if not matrix or not any(sum(row) for row in matrix):
            draw_text(self.screen, "Для тестовой матрицы пока мало сессий", rect.center, 17, COLORS["muted"], anchor="center")
            return
        test_accuracy = getattr(self.model_metrics, "test_accuracy", 0.0)
        macro_f1 = getattr(self.model_metrics, "macro_f1", 0.0)
        draw_text(self.screen, "МАТРИЦА ОШИБОК — НЕЗНАКОМЫЕ СЕССИИ", (rect.centerx, rect.top + 20), 17, bold=True, anchor="midtop")
        draw_text(
            self.screen,
            f"Тестовая точность {test_accuracy * 100:.1f}%   •   macro-F1 {macro_f1 * 100:.1f}%",
            (rect.centerx, rect.top + 48),
            14,
            COLORS["player"],
            anchor="midtop",
        )
        cell_size = 23
        grid = pygame.Rect(rect.left + 165, rect.top + 112, cell_size * 10, cell_size * 10)
        self.confusion_grid_rect = grid.copy()
        maximum = max(max(row) for row in matrix) or 1
        draw_text(self.screen, "ФАКТ", (grid.left - 52, grid.centery), 12, COLORS["muted"], anchor="center")
        draw_text(self.screen, "ПРОГНОЗ →", (grid.centerx, grid.top - 37), 12, COLORS["muted"], anchor="center")
        for index in range(10):
            draw_text(self.screen, str(index), (grid.left + index * cell_size + cell_size / 2, grid.top - 18), 11, COLORS["muted"], anchor="center")
            draw_text(self.screen, str(index), (grid.left - 15, grid.top + index * cell_size + cell_size / 2), 11, COLORS["muted"], anchor="center")
        for actual, row in enumerate(matrix):
            for predicted, value in enumerate(row):
                intensity = math.sqrt(value / maximum) if value else 0.0
                base = COLORS["health"] if actual == predicted else COLORS["enemy"]
                color = tuple(int(17 + (component - 17) * intensity) for component in base)
                cell = pygame.Rect(grid.left + predicted * cell_size, grid.top + actual * cell_size, cell_size, cell_size)
                logical_draw.rect(self.screen, color, cell)
                logical_draw.rect(self.screen, (50, 64, 84), cell, 1)
                if value:
                    draw_text(self.screen, str(value), cell.center, 10, COLORS["white"], bold=value == maximum, anchor="center")
        draw_text(self.screen, "Нажмите на красную ячейку, чтобы открыть момент повтора", (rect.centerx, rect.bottom - 29), 10, COLORS["warning"], anchor="center")
        draw_text(self.screen, "0 ожид. 1 сближ. 2 отход 3–4 обход 5 дальн. 6 ближн. 7 рывок 8 блок 9 лечение", (rect.centerx, rect.bottom - 13), 9, COLORS["muted"], anchor="center")

    def _open_confusion_example(self, position: tuple[int, int]) -> None:
        grid = self.confusion_grid_rect
        if not grid.collidepoint(position) or not self.model_metrics:
            return
        cell_size = grid.width / 10
        predicted = min(9, int((position[0] - grid.left) / cell_size))
        actual = min(9, int((position[1] - grid.top) / cell_size))
        examples = [
            item
            for item in getattr(self.model_metrics, "error_examples", ())
            if int(item.get("actual", -1)) == actual and int(item.get("predicted", -1)) == predicted
        ]
        if not examples:
            self.analysis_notice = "Для этой ячейки нет сохранённого ошибочного момента"
            return
        key = (actual, predicted)
        cursor = self.confusion_error_cursor.get(key, 0) % len(examples)
        self.confusion_error_cursor[key] = cursor + 1
        example = examples[cursor]
        replay = ReplayRecorder.load(DATA_DIR / f"replay_{example['session_id']}.json")
        if replay is None:
            self.analysis_notice = "Это старая сессия: её подробный повтор ещё не сохранялся"
            return
        self.analysis_replay_backup = self.replay_recorder
        self.replay_recorder = replay
        self.replay_arena = max(1, int(example.get("arena_id", 1)))
        self.replay_time = max(0.0, float(example.get("arena_time", 0.0)) - 1.2)
        self.replay_paused = False
        self.replay_return_state = "profile"
        self.replay_error_label = (
            f"Факт: {ACTION_LABELS_RU[PlayerAction(actual)]}  •  прогноз: {ACTION_LABELS_RU[PlayerAction(predicted)]}"
        )
        self.state = "replay"

    def _draw_profile(self) -> None:
        self.screen.fill(COLORS["background"])
        draw_text(self.screen, "ПОВЕДЕНЧЕСКИЙ ПРОФИЛЬ", (WIDTH / 2, 46), 38, COLORS["player"], bold=True, anchor="midtop")
        draw_text(self.screen, self.profile.style_name, (WIDTH / 2, 100), 28, COLORS["white"], bold=True, anchor="midtop")
        left = pygame.Rect(90, 170, 520, 390)
        right = pygame.Rect(670, 170, 520, 390)
        draw_panel(self.screen, left)
        draw_panel(self.screen, right)
        draw_text(self.screen, "Основные показатели", (left.left + 28, left.top + 24), 22, bold=True)
        metrics = [
            ("Агрессивность", self.profile.aggression, COLORS["enemy"]),
            ("Подвижность", self.profile.mobility, COLORS["accent"]),
            ("Защита", self.profile.defense, COLORS["health"]),
            ("Базовая предсказуемость", self.profile.predictability, COLORS["warning"]),
        ]
        for index, (label, value, color) in enumerate(metrics):
            metric_rect = pygame.Rect(left.left + 28, left.top + 78 + index * 70, left.width - 56, 52)
            draw_metric(self.screen, label, value, metric_rect, color)
        if self.model_metrics and hasattr(self.model_metrics, "confusion_matrix"):
            self._draw_confusion_matrix(right)
        else:
            draw_text(self.screen, "Распределение действий", (right.left + 28, right.top + 24), 22, bold=True)
            sorted_rates = sorted(self.profile.action_rates.items(), key=lambda item: item[1], reverse=True)[:6]
            max_rate = max((rate for _, rate in sorted_rates), default=1.0)
            for index, (action, rate) in enumerate(sorted_rates):
                y = right.top + 78 + index * 46
                draw_text(self.screen, ACTION_LABELS_RU[action], (right.left + 28, y), 16, COLORS["muted"])
                bar = pygame.Rect(right.left + 205, y + 3, 230, 12)
                draw_bar(self.screen, bar, rate, max_rate, COLORS["twin"])
                draw_text(self.screen, f"{rate * 100:.0f}%", (right.right - 28, y), 16, anchor="topright")
        quality_color = COLORS["health"] if self.data_quality.score >= 0.58 else COLORS["warning"]
        draw_text(
            self.screen,
            f"Данные: {self.data_quality.status} {self.data_quality.score * 100:.0f}%   •   примеров {self.profile.sample_count}   •   классов {self.data_quality.covered_classes}/10",
            (WIDTH / 2, 582),
            15,
            quality_color,
            bold=True,
            anchor="midtop",
        )
        draw_text(self.screen, self.analysis_notice or self.data_quality.warning, (WIDTH / 2, 608), 13, COLORS["muted"], anchor="midtop")
        profile_actions = ["ENTER  СОЗДАТЬ ДВОЙНИКА", "A  ПОЛНЫЙ AI-АНАЛИЗ"]
        for rect, label in zip(self._profile_action_rects(), profile_actions):
            self._draw_action_button(rect, label)
        if self.model_metrics and hasattr(self.model_metrics, "sequence_length"):
            parameters = f"{getattr(self.model_metrics, 'parameter_count', 0):,}".replace(",", " ")
            rl_label = " • offline RL принято" if getattr(self.model_metrics, "rl_applied", False) else ""
            model_text = f"GRU • 10 состояний • {parameters} параметров • {getattr(self.model_metrics, 'device', 'CPU')}{rl_label}"
        elif self.model_metrics:
            model_text = f"Резервная MLP 25–64–32–10 • точность {self.model_metrics.accuracy * 100:.1f}% • loss {self.model_metrics.loss:.3f}"
        else:
            model_text = "Используется внешняя модель прогнозирования"
        draw_text(self.screen, model_text, (WIDTH / 2, 707), 14, COLORS["player"], anchor="midbottom")

    def _draw_model_training(self) -> None:
        self.animated_background.draw(self.screen, 0.82)
        progress = self.ai_training_progress
        stage = str(progress.get("stage", "подготовка данных")).upper()
        epoch = int(progress.get("epoch", 0))
        total = max(1, int(progress.get("total", 1)))
        ratio = epoch / total if epoch else (0.12 + 0.08 * math.sin(self.ui_time * 2.4))
        draw_text(self.screen, "ОБУЧЕНИЕ ЦИФРОВОГО ДВОЙНИКА", (WIDTH / 2, 72), 40, COLORS["player"], bold=True, anchor="midtop")
        draw_text(self.screen, "Игра продолжает отрисовываться — обучение выполняется в фоне", (WIDTH / 2, 130), 18, COLORS["muted"], anchor="midtop")
        center = pygame.Vector2(WIDTH / 2, 310)
        for index in range(5):
            radius = 72 + index * 23
            start = self.ui_time * (0.7 + index * 0.11) + index
            logical_draw.arc(
                self.screen,
                COLORS["player"] if index % 2 == 0 else COLORS["twin"],
                pygame.Rect(center.x - radius, center.y - radius, radius * 2, radius * 2),
                start,
                start + 1.2 + ratio * 2.5,
                2,
            )
        draw_text(self.screen, stage, center, 22, COLORS["white"], bold=True, anchor="center")
        if epoch:
            draw_text(self.screen, f"Эпоха {epoch}/{total}", (center.x, center.y + 38), 17, COLORS["warning"], anchor="center")
        bar = pygame.Rect(280, 500, 720, 15)
        logical_draw.rect(self.screen, (25, 40, 61), bar, border_radius=8)
        logical_draw.rect(self.screen, COLORS["player"], (bar.x, bar.y, int(bar.width * max(0.0, min(1.0, ratio))), bar.height), border_radius=8)
        draw_text(
            self.screen,
            f"Качество данных: {self.data_quality.score * 100:.0f}%   •   покрыто действий: {self.data_quality.covered_classes}/10",
            (WIDTH / 2, 545),
            17,
            COLORS["health"] if self.data_quality.score >= 0.58 else COLORS["warning"],
            anchor="midtop",
        )
        draw_text(self.screen, self.data_quality.warning, (WIDTH / 2, 578), 14, COLORS["muted"], anchor="midtop")
        draw_text(self.screen, "После завершения автоматически откроется поведенческий профиль", (WIDTH / 2, 650), 15, COLORS["muted"], anchor="midtop")

    def _draw_line_chart(
        self,
        rect: pygame.Rect,
        values: list[float],
        color: tuple[int, int, int],
        title: str,
        maximum: float | None = None,
    ) -> None:
        logical_draw.rect(self.screen, (10, 17, 30), rect, border_radius=9)
        logical_draw.rect(self.screen, (55, 75, 103), rect, 1, border_radius=9)
        draw_text(self.screen, title, (rect.left + 14, rect.top + 10), 15, COLORS["muted"], bold=True)
        if not values:
            draw_text(self.screen, "Нет истории", rect.center, 15, COLORS["muted"], anchor="center")
            return
        top = max(maximum or 0.0, max(values), 1e-6)
        chart = rect.inflate(-28, -58)
        chart.y += 14
        points = [
            (
                chart.left + index / max(1, len(values) - 1) * chart.width,
                chart.bottom - max(0.0, min(1.0, value / top)) * chart.height,
            )
            for index, value in enumerate(values)
        ]
        if len(points) > 1:
            logical_draw.lines(self.screen, color, False, points, 3)
        for point in points:
            logical_draw.circle(self.screen, color, point, 3)
        draw_text(self.screen, f"{values[-1]:.3f}", (rect.right - 12, rect.top + 9), 14, color, bold=True, anchor="topright")

    def _draw_model_analysis(self) -> None:
        self.screen.fill(COLORS["background"])
        self._draw_background_nodes()
        draw_text(self.screen, "AI-ЛАБОРАТОРИЯ ДВОЙНИКА", (WIDTH / 2, 26), 34, COLORS["player"], bold=True, anchor="midtop")
        if not self.model_metrics:
            draw_text(self.screen, "Модель ещё не обучена", (WIDTH / 2, HEIGHT / 2), 24, COLORS["muted"], anchor="center")
            return
        history = list(getattr(self.model_metrics, "training_history", ()))
        self._draw_line_chart(pygame.Rect(55, 95, 370, 210), [float(item["loss"]) for item in history], COLORS["enemy"], "LOSS")
        self._draw_line_chart(pygame.Rect(455, 95, 370, 210), [float(item["accuracy"]) for item in history], COLORS["health"], "ACCURACY", 1.0)
        self._draw_line_chart(pygame.Rect(855, 95, 370, 210), [float(item["macro_f1"]) for item in history], COLORS["accent"], "MACRO-F1", 1.0)
        comparison = pygame.Rect(55, 335, 550, 245)
        draw_panel(self.screen, comparison)
        draw_text(self.screen, "ОБЩАЯ МОДЕЛЬ → ПЕРСОНАЛЬНАЯ", (comparison.centerx, comparison.top + 20), 19, bold=True, anchor="midtop")
        general = float(getattr(self.model_metrics, "general_player_accuracy", 0.0))
        personal = float(self.model_metrics.accuracy)
        for index, (label, value, color) in enumerate((("До адаптации", general, COLORS["muted"]), ("После адаптации", personal, COLORS["player"]))):
            y = comparison.top + 82 + index * 72
            draw_text(self.screen, label, (comparison.left + 25, y), 16, COLORS["white"], bold=True)
            bar = pygame.Rect(comparison.left + 180, y + 4, 300, 18)
            draw_bar(self.screen, bar, value, 1.0, color)
            draw_text(self.screen, f"{value * 100:.1f}%", (comparison.right - 24, y), 16, color, bold=True, anchor="topright")
        sessions_panel = pygame.Rect(635, 335, 590, 245)
        draw_panel(self.screen, sessions_panel)
        player_history = [item for item in self.metrics_history if item.get("player_id") == self.current_player.player_id]
        self._draw_line_chart(
            sessions_panel.inflate(-24, -40).move(0, 14),
            [float(item.get("personal_accuracy", 0.0)) for item in player_history],
            COLORS["warning"],
            "ТОЧНОСТЬ ПО СЕССИЯМ",
            1.0,
        )
        draw_text(
            self.screen,
            f"Игрок: {self.current_player.name}   •   качество данных {self.data_quality.score * 100:.0f}%   •   реальных наград {self.data_quality.real_reward_ratio * 100:.0f}%",
            (WIDTH / 2, 607),
            15,
            COLORS["muted"],
            anchor="midtop",
        )
        draw_text(self.screen, "ESC — вернуться к профилю", (WIDTH / 2, 668), 17, COLORS["warning"], bold=True, anchor="midtop")

    def _draw_result(self) -> None:
        self.screen.fill(COLORS["background"])
        self._draw_background_nodes()
        draw_text(self.screen, "ДВОЙНИК ПОБЕЖДЁН", (WIDTH / 2, 75), 49, COLORS["health"], bold=True, anchor="midtop")
        draw_text(self.screen, "Чтобы победить копию, ты изменил собственные привычки.", (WIDTH / 2, 145), 22, anchor="midtop")
        panel = pygame.Rect(70, 205, 1140, 350)
        draw_panel(self.screen, panel)
        draw_text(self.screen, "Итог сессии", (panel.left + 34, panel.top + 25), 25, bold=True)
        items = [
            ("Поведенческий стиль", self.profile.style_name),
            ("Главная привычка", ACTION_LABELS_RU[self.profile.favorite_action]),
            ("Обучающих примеров", str(self.profile.sample_count)),
            (
                "Персональная / тестовая точность",
                (
                    f"{self.model_metrics.accuracy * 100:.1f}% / {getattr(self.model_metrics, 'test_accuracy', self.model_metrics.accuracy) * 100:.1f}%"
                    if self.model_metrics
                    else "внешняя модель"
                ),
            ),
            ("Точность прогнозов в финале", f"{self.synchronization * 100:.1f}%"),
            ("Изменений тактики в финале", str(self.twin_action_changes)),
        ]
        for index, (label, value) in enumerate(items):
            y = panel.top + 80 + index * 46
            draw_text(self.screen, label, (panel.left + 34, y), 17, COLORS["muted"])
            draw_text(self.screen, value, (panel.left + 500, y), 17, COLORS["white"], bold=True, anchor="topright")
        draw_text(self.screen, "Тепловая карта перемещений", (panel.left + 570, panel.top + 27), 21, bold=True)
        heatmap_rect = pygame.Rect(panel.left + 570, panel.top + 74, 520, 226)
        self._draw_heatmap(heatmap_rect)
        draw_text(self.screen, "редко", (heatmap_rect.left, heatmap_rect.bottom + 10), 14, COLORS["muted"])
        draw_text(self.screen, "часто", (heatmap_rect.right, heatmap_rect.bottom + 10), 14, COLORS["player"], anchor="topright")
        button_labels = ["R  НОВАЯ СЕССИЯ", "V  ПОВТОР", "L  ЛАБОРАТОРИЯ", "ESC  МЕНЮ"]
        for rect, label in zip(self._result_button_rects(), button_labels):
            self._draw_action_button(rect, label)
        average_confidence = sum(self.prediction_confidences) / len(self.prediction_confidences) if self.prediction_confidences else 0.0
        analysis_line = f"Живой анализ: точность {self.synchronization * 100:.1f}% • средняя уверенность {average_confidence * 100:.1f}% • ошибок {len(self.sync_history) - sum(self.sync_history)}"
        draw_text(self.screen, analysis_line, (WIDTH / 2, 635), 15, COLORS["player"], anchor="midtop")
        if self.saved_paths:
            draw_text(self.screen, f"Данные сохранены: data/{self.saved_paths[0].name}", (WIDTH / 2, 665), 14, COLORS["muted"], anchor="midtop")

    def _draw_replay(self) -> None:
        self.screen.fill(COLORS["background"])
        arena = Arena(self.replay_arena)
        arena.draw(self.screen)
        frames = self.replay_recorder.frames_for_arena(self.replay_arena)
        if frames:
            points = [
                pygame.Vector2(
                    arena.bounds.left + frame.normalized_x * arena.bounds.width,
                    arena.bounds.top + frame.normalized_y * arena.bounds.height,
                )
                for frame in frames
                if frame.arena_time <= self.replay_time
            ]
            if len(points) > 1:
                layer = self.alpha_surface
                layer.fill((0, 0, 0, 0))
                logical_draw.lines(layer, (*COLORS["twin"], 75), False, points[-180:], 3)
                logical_blit(self.screen, layer, (0, 0))
            position = self.replay_recorder.position_at(self.replay_arena, self.replay_time, arena.bounds)
            if position:
                logical_draw.circle(self.screen, COLORS["player"], position, 18)
                logical_draw.circle(self.screen, COLORS["twin"], position, 27, 2)
        logical_draw.rect(self.screen, COLORS["panel"], (0, 0, WIDTH, 80))
        draw_text(self.screen, "ПОВТОР СЕССИИ", (28, 19), 26, COLORS["player"], bold=True)
        draw_text(self.screen, f"Арена {self.replay_arena}   •   {self.replay_time:05.1f} сек.", (WIDTH / 2, 22), 20, COLORS["white"], bold=True, anchor="midtop")
        draw_text(self.screen, "1–4 арена   SPACE пауза   ESC назад", (WIDTH - 28, 22), 16, COLORS["muted"], anchor="topright")
        if self.replay_error_label:
            draw_text(self.screen, self.replay_error_label, (WIDTH / 2, 88), 17, COLORS["enemy"], bold=True, anchor="midtop")
        events = [event for event in self.replay_recorder.events if event.arena_id == self.replay_arena and abs(event.arena_time - self.replay_time) < 1.2]
        if events:
            draw_text(self.screen, events[-1].name, (WIDTH / 2, 118 if self.replay_error_label else 98), 20, COLORS["warning"], bold=True, anchor="midtop")
        if not frames:
            draw_text(self.screen, "Для этой арены нет записи", (WIDTH / 2, HEIGHT / 2), 28, COLORS["muted"], anchor="center")

    def _draw_lab(self) -> None:
        self.screen.fill(COLORS["background"])
        draw_text(self.screen, "ЛАБОРАТОРИЯ ПОВЕДЕНИЯ", (WIDTH / 2, 35), 37, COLORS["player"], bold=True, anchor="midtop")
        draw_text(self.screen, "Сравнение двух последних сохранённых сессий", (WIDTH / 2, 85), 18, COLORS["muted"], anchor="midtop")
        if not self.lab_sessions:
            draw_text(self.screen, "Сохранённых сессий пока нет", (WIDTH / 2, HEIGHT / 2), 28, COLORS["muted"], anchor="center")
            draw_text(self.screen, "ESC — меню", (WIDTH / 2, 650), 18, COLORS["warning"], anchor="midtop")
            return
        sessions = self.lab_sessions[self.lab_index : self.lab_index + 2]
        colors = [COLORS["player"], COLORS["twin"]]
        for index, session in enumerate(sessions):
            rect = pygame.Rect(80 + index * 580, 135, 540, 480)
            draw_panel(self.screen, rect)
            draw_text(self.screen, f"Игрок {session.player_id}  •  сессия {session.session_id}", (rect.left + 25, rect.top + 20), 17, colors[index], bold=True)
            draw_text(self.screen, f"Примеров: {session.sample_count}  •  Главное: {session.favorite_action}", (rect.left + 25, rect.top + 54), 15, COLORS["muted"])
            heat_rect = pygame.Rect(rect.right - 155, rect.top + 18, 125, 58)
            self._draw_session_heatmap(session, heat_rect, colors[index])
            maximum = max((session.rate(action) for action in PlayerAction), default=1.0)
            for action_index, action in enumerate(PlayerAction):
                y = rect.top + 95 + action_index * 34
                draw_text(self.screen, ACTION_LABELS_RU[action], (rect.left + 24, y), 14, COLORS["muted"])
                bar = pygame.Rect(rect.left + 185, y + 3, 270, 10)
                draw_bar(self.screen, bar, session.rate(action), max(0.01, maximum), colors[index])
                draw_text(self.screen, f"{session.rate(action) * 100:.0f}%", (rect.right - 22, y), 13, anchor="topright")
        if len(sessions) == 1:
            draw_text(self.screen, "Пройдите ещё одну сессию для сравнения", (920, 360), 20, COLORS["muted"], anchor="center")
        draw_text(self.screen, "← → выбрать пару сессий   •   ESC назад", (WIDTH / 2, 665), 18, COLORS["warning"], anchor="midtop")

    def _draw_session_heatmap(self, session: SessionSummary, rect: pygame.Rect, color: tuple[int, int, int]) -> None:
        maximum = max((max(row) for row in session.heatmap), default=1) or 1
        cell_width = rect.width / 12
        cell_height = rect.height / 6
        logical_draw.rect(self.screen, (9, 15, 27), rect)
        for row_index, row in enumerate(session.heatmap):
            for column_index, value in enumerate(row):
                if not value:
                    continue
                intensity = math.sqrt(value / maximum)
                cell_color = tuple(int(18 + (component - 18) * intensity) for component in color)
                logical_draw.rect(
                    self.screen,
                    cell_color,
                    (
                        rect.left + column_index * cell_width,
                        rect.top + row_index * cell_height,
                        math.ceil(cell_width) + 1,
                        math.ceil(cell_height) + 1,
                    ),
                )
        if len(session.health_series) > 1:
            step = max(1, len(session.health_series) // max(2, rect.width))
            sampled = session.health_series[::step]
            points = [
                (
                    rect.left + index / max(1, len(sampled) - 1) * rect.width,
                    rect.bottom - max(0.0, min(1.0, value)) * rect.height,
                )
                for index, value in enumerate(sampled)
            ]
            if len(points) > 1:
                logical_draw.lines(self.screen, COLORS["warning"], False, points, 1)
        logical_draw.rect(self.screen, (72, 91, 118), rect, 1)

    def _draw_tournament(self) -> None:
        self.screen.fill(COLORS["background"])
        self._draw_background_nodes()
        draw_text(self.screen, "ТУРНИР ЦИФРОВЫХ ДВОЙНИКОВ", (WIDTH / 2, 38), 36, COLORS["warning"], bold=True, anchor="midtop")
        if not self.tournament:
            return
        first, second = self.tournament.first, self.tournament.second
        left_position = pygame.Vector2(330, 370)
        right_position = pygame.Vector2(950, 370)
        logical_draw.circle(self.screen, COLORS["player"], left_position, 42)
        logical_draw.circle(self.screen, COLORS["twin"], right_position, 42)
        logical_draw.line(self.screen, COLORS["muted"], left_position + pygame.Vector2(50, 0), right_position - pygame.Vector2(50, 0), 2)
        draw_text(self.screen, first.name, (left_position.x, 180), 20, COLORS["player"], bold=True, anchor="midtop")
        draw_text(self.screen, second.name, (right_position.x, 180), 20, COLORS["twin"], bold=True, anchor="midtop")
        draw_bar(self.screen, pygame.Rect(170, 225, 320, 18), first.health, 100, COLORS["health"], "ЗДОРОВЬЕ")
        draw_bar(self.screen, pygame.Rect(790, 225, 320, 18), second.health, 100, COLORS["health"], "ЗДОРОВЬЕ")
        draw_text(self.screen, ACTION_LABELS_RU[self.tournament.last_first], (left_position.x, 445), 18, COLORS["muted"], anchor="midtop")
        draw_text(self.screen, ACTION_LABELS_RU[self.tournament.last_second], (right_position.x, 445), 18, COLORS["muted"], anchor="midtop")
        draw_text(self.screen, f"{self.tournament.round_time:04.1f} сек.", (WIDTH / 2, 280), 24, COLORS["white"], bold=True, anchor="midtop")
        if self.tournament.finished:
            draw_text(self.screen, f"ПОБЕДИТЕЛЬ: {self.tournament.winner.name}", (WIDTH / 2, 545), 27, COLORS["warning"], bold=True, anchor="midtop")
        else:
            draw_text(self.screen, "Двойники используют распределения действий реальных сессий", (WIDTH / 2, 545), 18, COLORS["muted"], anchor="midtop")
        draw_text(self.screen, "R новый бой   •   ESC меню", (WIDTH / 2, 645), 19, COLORS["warning"], anchor="midtop")


if __name__ == "__main__":
    DigitalTwinGame().run()
