import csv
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from digital_twin_game.actions import PlayerAction
from digital_twin_game.analysis import ComboTracker, PositionHeatmap
from digital_twin_game.abilities import AbilitySystem, AbilityType
from digital_twin_game.arena import Arena
from digital_twin_game.behavior import BehaviorProfile
from digital_twin_game.data import ActionOutcome, GameplayDataCollector, TrainingSample
from digital_twin_game.data_quality import analyze_data_quality
from digital_twin_game.calibration import CalibrationChallenge
from digital_twin_game.entities import Enemy, Player
from digital_twin_game.editor import ArenaEditor, ArenaLayout
from digital_twin_game.features import build_feature_vector
from digital_twin_game.game import DigitalTwinGame
from digital_twin_game.hidpi import draw as logical_draw, register_surface, unregister_surface
from digital_twin_game.lab import load_sessions
from digital_twin_game.model_interface import ACTION_COUNT, FEATURE_NAMES, PlaceholderPredictor
from digital_twin_game.neural_model import NeuralActionModel
from digital_twin_game.objectives import ObjectiveType, TrainingObjective
from digital_twin_game.players import PlayerRegistry
from digital_twin_game.replay import ReplayRecorder
from digital_twin_game.sequence_data import (
    SEQUENCE_LENGTH,
    SessionDataset,
    balanced_sample_weights,
    build_sequences,
    load_session_datasets,
    split_by_player,
    split_by_session,
)
from digital_twin_game.torch_model import TorchSequenceModel, torch_available
from digital_twin_game.tournament import TournamentFighter, TournamentMatch
from digital_twin_game.viewport import DisplayViewport, SUPPORTED_DISPLAY_SIZES
from digital_twin_game.weapons import WeaponType


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_fixed_dimensions(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 25)
        self.assertEqual(ACTION_COUNT, 10)
        self.assertEqual(len(PlayerAction), 10)

    def test_all_requested_display_sizes_preserve_aspect_ratio(self) -> None:
        for display_size in SUPPORTED_DISPLAY_SIZES:
            with self.subTest(display_size=display_size):
                viewport = DisplayViewport((1280, 720), display_size)
                self.assertLessEqual(viewport.rect.width, display_size[0])
                self.assertLessEqual(viewport.rect.height, display_size[1])
                self.assertAlmostEqual(viewport.rect.width / viewport.rect.height, 16 / 9, places=2)
                display_center = viewport.logical_to_display((640, 360))
                logical_center = viewport.display_to_logical(display_center)
                self.assertLessEqual(abs(logical_center[0] - 640), 1)
                self.assertLessEqual(abs(logical_center[1] - 360), 1)

    def test_letterbox_does_not_activate_ui_outside_game_area(self) -> None:
        viewport = DisplayViewport((1280, 720), (1024, 768))
        self.assertGreater(viewport.rect.top, 0)
        self.assertEqual(viewport.display_to_logical((10, 10)), (-10_000, -10_000))
        self.assertEqual(viewport.display_to_logical((10, 10), clamp=True)[1], 0)

    def test_hidpi_drawing_uses_native_surface_pixels(self) -> None:
        surface = register_surface(pygame.Surface((256, 144)), 2.0)
        physical_rect = logical_draw.rect(surface, (255, 255, 255), (10, 12, 30, 16))
        self.assertEqual(physical_rect, pygame.Rect(20, 24, 60, 32))
        self.assertEqual(surface.get_at((25, 30))[:3], (255, 255, 255))
        unregister_surface(surface)

    def test_game_renders_directly_at_viewport_resolution(self) -> None:
        game = DigitalTwinGame()
        self.assertEqual(game.screen.get_size(), game.viewport.rect.size)
        self.assertAlmostEqual(game.render_scale, game.viewport.rect.width / 1280)

    def test_selection_continue_buttons_are_clickable(self) -> None:
        game = DigitalTwinGame()
        game.state = "weapon_select"
        weapon_click = game.viewport.logical_to_display(game._weapon_continue_rect().center)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": weapon_click}))
        game._handle_events()
        self.assertEqual(game.state, "loadout")

        game.loadout_selection = set(list(AbilityType)[:3])
        loadout_click = game.viewport.logical_to_display(game._loadout_continue_rect().center)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": loadout_click}))
        game._handle_events()
        self.assertEqual(game.state, "training")

    def test_player_name_accepts_text_input(self) -> None:
        game = DigitalTwinGame()
        game._open_player_select()
        pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, {"text": "Игрок"}))
        game._handle_events()
        self.assertEqual(game.player_name_input, "Игрок")

    def test_real_neural_model_has_expected_architecture(self) -> None:
        model = NeuralActionModel(seed=1)
        self.assertEqual(model.LAYERS, (25, 64, 32, 10))
        self.assertEqual(model.parameter_count, 4074)
        prediction = model.predict([0.0] * 25)
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0, places=6)

    def test_neural_model_save_and_load(self) -> None:
        model = NeuralActionModel(seed=7)
        expected = model.predict([0.1] * 25).probabilities
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = NeuralActionModel.load(path)
        actual = loaded.predict([0.1] * 25).probabilities
        for first, second in zip(expected, actual):
            self.assertAlmostEqual(first, second)

    def test_feature_vector_is_normalized(self) -> None:
        arena = Arena(1)
        player = Player((180, 360))
        enemy = Enemy((1050, 360))
        features = build_feature_vector(player, enemy, arena)
        self.assertEqual(len(features), 25)
        self.assertTrue(all(-1 <= value <= 1 for value in features))

    def test_placeholder_obeys_probability_contract(self) -> None:
        predictor = PlaceholderPredictor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        prediction = predictor.predict([0] * 25)
        self.assertEqual(len(prediction.probabilities), 10)
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0)

    def test_collector_writes_model_ready_csv(self) -> None:
        collector = GameplayDataCollector(0.2)
        collector.update(0.2, 1, [0] * 25, PlayerAction.DASH)
        with tempfile.TemporaryDirectory() as directory:
            paths = collector.save(Path(directory))
            self.assertIsNotNone(paths)
            csv_path, meta_path = paths
            self.assertTrue(meta_path.exists())
            with csv_path.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["action_name"], "DASH")
            self.assertEqual(len([name for name in FEATURE_NAMES if name in rows[0]]), 25)

    def test_rare_event_is_recorded_without_waiting_for_interval(self) -> None:
        collector = GameplayDataCollector(0.2)
        recorded = collector.update(0.01, 1, [0.0] * 25, PlayerAction.DASH, force=True)
        self.assertTrue(recorded)
        self.assertEqual(collector.samples[0].action_id, PlayerAction.DASH)

    def test_real_outcome_reward_round_trip(self) -> None:
        collector = GameplayDataCollector(0.2, "player_test")
        collector.update(0.01, 2, [0.0] * 25, PlayerAction.DASH, force=True, arena_time=4.2)
        collector.record_outcome(PlayerAction.DASH, ActionOutcome(perfect_dodge=1, damage_taken=2.0))
        with tempfile.TemporaryDirectory() as directory:
            collector.save(Path(directory))
            loaded = load_session_datasets(Path(directory))[0]
        sample = loaded.samples[0]
        self.assertEqual(loaded.player_id, "player_test")
        self.assertEqual(sample.perfect_dodge, 1)
        self.assertGreater(sample.reward, 0.0)
        self.assertEqual(sample.arena_time, 4.2)

    def test_player_registry_persists_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "players.json"
            registry = PlayerRegistry.load(path)
            created = registry.create("Алекс")
            registry.record_session(created.player_id)
            loaded = PlayerRegistry.load(path)
        self.assertEqual(loaded.active.name, "Алекс")
        self.assertEqual(loaded.active.session_count, 1)

    def test_player_registry_renames_and_deletes_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "players.json"
            registry = PlayerRegistry.load(path)
            original_id = registry.active.player_id
            second = registry.create("Второй игрок")
            renamed = registry.rename(second.player_id, "Новое имя")
            self.assertEqual(renamed.name, "Новое имя")
            active = registry.delete(second.player_id)
            self.assertEqual(active.player_id, original_id)
            self.assertEqual(len(registry.profiles), 1)
            with self.assertRaises(ValueError):
                registry.delete(original_id)

    def test_quality_report_lists_missing_actions(self) -> None:
        collector = GameplayDataCollector(0.1)
        for _ in range(10):
            collector.update(0.1, 1, [0.0] * 25, PlayerAction.IDLE)
        report = analyze_data_quality(collector.samples)
        self.assertEqual(report.covered_classes, 1)
        self.assertIn(PlayerAction.DASH, report.missing_actions)

    def test_calibration_challenge_tracks_rare_actions(self) -> None:
        challenge = CalibrationChallenge(2)
        for _ in range(4):
            challenge.record("dash")
        challenge.record("perfect_dodge")
        self.assertTrue(challenge.complete)

    @staticmethod
    def _temporal_session(session_id: str, marker: float, count: int = 12) -> SessionDataset:
        samples = []
        for index in range(count):
            features = [0.0] * 25
            features[0] = marker
            samples.append(
                TrainingSample(
                    timestamp=float(index),
                    session_id=session_id,
                    arena_id=1,
                    features=tuple(features),
                    action_id=index % 2,
                )
            )
        return SessionDataset(session_id, f"player_{session_id}", tuple(samples))

    def test_temporal_windows_never_cross_sessions(self) -> None:
        examples = build_sequences(
            [self._temporal_session("first", -0.5), self._temporal_session("second", 0.5)]
        )
        self.assertTrue(all(len(example.states) == SEQUENCE_LENGTH for example in examples))
        for example in examples:
            expected = -0.5 if example.session_id == "first" else 0.5
            self.assertTrue(all(state[0] == expected for state in example.states))

    def test_session_split_has_no_overlap(self) -> None:
        sessions = [self._temporal_session(str(index), index / 20) for index in range(10)]
        split = split_by_session(sessions)
        groups = [set(item.session_id for item in group) for group in (split.train, split.validation, split.test)]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(len(set.union(*groups)), 10)

    def test_player_split_keeps_all_player_sessions_together(self) -> None:
        sessions = []
        for player in range(5):
            for session_index in range(2):
                source = self._temporal_session(f"{player}_{session_index}", player / 10)
                sessions.append(SessionDataset(source.session_id, f"player_{player}", source.samples))
        split = split_by_player(sessions)
        player_groups = [set(item.player_id for item in group) for group in (split.train, split.validation, split.test)]
        self.assertFalse(player_groups[0] & player_groups[1])
        self.assertFalse(player_groups[0] & player_groups[2])
        self.assertFalse(player_groups[1] & player_groups[2])

    def test_balancing_equalizes_present_class_weight(self) -> None:
        session = self._temporal_session("imbalanced", 0.0, 10)
        examples = list(build_sequences([session]))
        examples = examples[:9]
        # Create an 8:1 imbalance while preserving valid temporal examples.
        examples = [examples[0]] * 8 + [examples[1]]
        weights = balanced_sample_weights(examples)
        totals = [sum(weight for weight, example in zip(weights, examples) if example.action_id == action) for action in (0, 1)]
        self.assertLess(max(totals) / min(totals), 4.0)

    @unittest.skipUnless(torch_available(), "PyTorch is not installed")
    def test_gru_keeps_ten_states_and_round_trips(self) -> None:
        import torch

        model = TorchSequenceModel(device=torch.device("cpu"))
        for index in range(14):
            prediction = model.predict([min(1.0, index / 20)] + [0.0] * 24)
        self.assertEqual(len(model._history), 10)
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0, places=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gru.pt"
            model.save(path)
            loaded = TorchSequenceModel.load(path, torch.device("cpu"))
        self.assertEqual(loaded.parameter_count, 44892)

    def test_profile_uses_samples(self) -> None:
        collector = GameplayDataCollector(0.1)
        for _ in range(5):
            collector.update(0.1, 1, [0] * 25, PlayerAction.RANGED_ATTACK)
        profile = BehaviorProfile.from_samples(collector.samples)
        self.assertEqual(profile.favorite_action, PlayerAction.RANGED_ATTACK)
        self.assertEqual(profile.sample_count, 5)

    def test_combo_tracker_recognises_combat_sequence(self) -> None:
        tracker = ComboTracker()
        tracker.record(0.0, PlayerAction.RANGED_ATTACK)
        tracker.record(0.2, PlayerAction.APPROACH)  # movement does not break the combo
        tracker.record(0.5, PlayerAction.DASH)
        result = tracker.record(0.8, PlayerAction.MELEE_ATTACK)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "РАЗРЫВ ШАБЛОНА")

    def test_heatmap_records_position(self) -> None:
        heatmap = PositionHeatmap(columns=4, rows=2)
        bounds = pygame.Rect(0, 0, 400, 200)
        heatmap.add(pygame.Vector2(350, 150), bounds)
        self.assertEqual(heatmap.sample_count, 1)
        self.assertEqual(heatmap.cells[1][3], 1)

    def test_enemy_telegraphs_before_melee_damage(self) -> None:
        arena = Arena(1)
        player = Player((200, 400))
        enemy = Enemy((250, 400), "hunter")
        enemy.attack_cooldown = 0
        projectiles = []
        original_health = player.health
        enemy.update(0.01, player, arena, projectiles)
        self.assertGreater(enemy.attack_windup, 0)
        self.assertEqual(player.health, original_health)
        enemy.update(0.6, player, arena, projectiles)
        self.assertLess(player.health, original_health)

    def test_expired_timer_waits_for_remaining_enemies(self) -> None:
        game = DigitalTwinGame()
        game._start_session()
        game.round_time_left = 0.01
        game._update_combat(0.02)
        self.assertEqual(game.arena_id, 1)
        self.assertTrue(any(enemy.alive for enemy in game.enemies))

        game.objective.progress = game.objective.target
        for enemy in game.enemies:
            enemy.health = 0
        for _ in range(35):
            game._update_combat(0.033)
        self.assertEqual(game.state, "upgrade")
        game._choose_upgrade(0)
        self.assertEqual(game.arena_id, 2)

    def test_ability_system_enforces_cooldown(self) -> None:
        system = AbilitySystem([AbilityType.TRAP, AbilityType.WAVE, AbilityType.SHIELD])
        self.assertEqual(system.activate(0), AbilityType.TRAP)
        self.assertIsNone(system.activate(0))
        system.update(6.1)
        self.assertEqual(system.activate(0), AbilityType.TRAP)

    def test_training_objectives_have_distinct_types(self) -> None:
        bounds = pygame.Rect(50, 100, 1000, 500)
        self.assertEqual(TrainingObjective(1, bounds).kind, ObjectiveType.HOLD)
        self.assertEqual(TrainingObjective(2, bounds).kind, ObjectiveType.COLLECT)
        self.assertEqual(TrainingObjective(3, bounds).kind, ObjectiveType.PROTECT)

    def test_replay_round_trip(self) -> None:
        recorder = ReplayRecorder()
        bounds = pygame.Rect(0, 0, 100, 100)
        recorder.update(0.1, 1, 0.1, pygame.Vector2(25, 75), bounds, PlayerAction.DASH)
        recorder.add_event(1, 0.1, "Тест")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            recorder.save(path)
            loaded = ReplayRecorder.load(path)
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.frames), 1)
        self.assertEqual(loaded.events[0].name, "Тест")

    def test_editor_layout_round_trip(self) -> None:
        layout = ArenaLayout(
            [(100, 100, 80, 40)],
            [(300, 300)],
            (150, 350),
            [(900, 350, "sniper")],
            "hold",
            (640, 420),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arena.json"
            layout.save(path)
            loaded = ArenaLayout.load(path)
        self.assertEqual(loaded.obstacles, layout.obstacles)
        self.assertEqual(loaded.enemy_spawns, layout.enemy_spawns)
        self.assertEqual(loaded.game_mode, "hold")
        self.assertEqual(loaded.objective_position, (640, 420))

    def test_editor_rotates_selected_wall(self) -> None:
        editor = ArenaEditor(ArenaLayout(obstacles=[(200, 200, 160, 40)]))
        editor.selected_obstacle = 0
        self.assertTrue(editor.rotate_selected_wall())
        rotated = pygame.Rect(editor.layout.obstacles[0])
        self.assertEqual(rotated.size, (40, 160))
        self.assertEqual(rotated.center, pygame.Rect(200, 200, 160, 40).center)

    def test_destructible_object_takes_damage(self) -> None:
        arena = Arena(1)
        target = arena.destructibles[0]
        original_health = target.health
        hit = arena.hit_destructible(pygame.Vector2(target.rect.left - 10, target.rect.centery), pygame.Vector2(target.rect.right + 10, target.rect.centery), 12)
        self.assertIsNotNone(hit)
        self.assertLess(target.health, original_health)

    def test_shotgun_emits_five_projectiles(self) -> None:
        player = Player((200, 300))
        player.set_weapon(WeaponType.SHOTGUN)
        projectiles = player.fire(pygame.Vector2(500, 300))
        self.assertEqual(len(projectiles), 5)

    def test_main_menu_has_three_actions(self) -> None:
        game = DigitalTwinGame()
        self.assertEqual(len(game._menu_button_rects()), 3)
        game._activate_menu_option(2)
        self.assertEqual(game.state, "tutorial")

    def test_start_game_opens_player_selection(self) -> None:
        game = DigitalTwinGame()
        game._activate_menu_option(0)
        self.assertEqual(game.state, "player_select")
        game._confirm_player_selection()
        self.assertEqual(game.state, "weapon_select")

    def test_player_launcher_uses_lightweight_binary_dependency(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        requirements = (project_root / "requirements-player.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual([line for line in requirements if line.strip()], ["pygame==2.6.1"])
        launcher = (project_root / "start_game.bat").read_text(encoding="utf-8")
        self.assertIn("--only-binary=:all:", launcher)
        self.assertIn('PYTHON_ARG=-3.10', launcher)
        self.assertIn("(3, 10) <= sys.version_info[:2] <= (3, 13)", launcher)
        self.assertIn("Python 3.14 is not supported", launcher)
        self.assertNotIn("pip install -r requirements.txt", launcher)

    def test_interactive_tutorial_starts_after_pages(self) -> None:
        game = DigitalTwinGame()
        game._start_interactive_tutorial()
        self.assertEqual(game.state, "tutorial_play")
        self.assertEqual(game.tutorial_step, 0)
        self.assertEqual(len(game.enemies), 1)

    def test_upgrade_applies_on_next_arena(self) -> None:
        game = DigitalTwinGame()
        game._start_session()
        game._advance_training()
        self.assertEqual(game.state, "upgrade")
        game._choose_upgrade(1)
        self.assertEqual(game.arena_id, 2)
        self.assertGreater(game.player.damage_multiplier, 1.0)

    def test_demo_mode_shortens_calibration(self) -> None:
        game = DigitalTwinGame()
        game.demo_mode = True
        game._start_session()
        self.assertEqual(game.round_time_left, 14.0)

    def test_loadout_bars_stay_full_and_reset_on_change(self) -> None:
        game = DigitalTwinGame()
        for _ in range(200):
            game._update_loadout_animation(0.01)
        selected_indices = [index for index, ability in enumerate(AbilityType) if ability in game.loadout_selection]
        self.assertTrue(all(game.loadout_bar_progress[index] == 1.0 for index in selected_indices))
        game._update_loadout_animation(5.0)
        self.assertTrue(all(game.loadout_bar_progress[index] == 1.0 for index in selected_indices))
        game._toggle_loadout_ability(selected_indices[0])
        self.assertTrue(all(progress == 0.0 for progress in game.loadout_bar_progress))

    def test_tournament_reaches_result(self) -> None:
        first = TournamentFighter.demo("A", True)
        second = TournamentFighter.demo("B", False)
        match = TournamentMatch(first, second)
        for _ in range(1000):
            match.update(0.1)
            if match.finished:
                break
        self.assertTrue(match.finished)
        self.assertIn(match.winner, (first, second))

    def test_shield_enemy_blocks_front_damage(self) -> None:
        enemy = Enemy((500, 350), "shield")
        enemy.facing = pygame.Vector2(-1, 0)
        front = enemy.take_damage(20, pygame.Vector2(400, 350))
        back = enemy.take_damage(20, pygame.Vector2(600, 350))
        self.assertLess(front, back)

    def test_laboratory_loads_session_heatmap(self) -> None:
        collector = GameplayDataCollector(0.1)
        features = [0.0] * 25
        features[15] = 0.7
        features[17] = 0.4
        collector.update(0.1, 1, features, PlayerAction.APPROACH)
        with tempfile.TemporaryDirectory() as directory:
            collector.save(Path(directory))
            sessions = load_sessions(Path(directory))
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sum(sum(row) for row in sessions[0].heatmap), 1)

    def test_copier_steals_last_used_ability(self) -> None:
        player = Player((200, 300))
        player.last_used_ability = AbilityType.TRAP.value
        copier = Enemy((245, 300), "copier")
        copier.queued_attack = "melee"
        copier._release_attack(player, [], 0.0)
        self.assertEqual(copier.copied_ability, AbilityType.TRAP.value)
        self.assertGreater(player.ability_lock_timer, 0)


if __name__ == "__main__":
    unittest.main()
