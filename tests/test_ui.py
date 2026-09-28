"""UI callback contracts; run with the app's Gradio environment."""

import asyncio
import importlib.util
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio is not installed")
class GuidedUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch.dict("os.environ", {"PIPER_DATA_DIR": self.temp.name}):
            from app import main
        from app.projects import ProjectStore
        from app.training import TrainingJobs
        self.ui = main
        self.store = ProjectStore(Path(self.temp.name))
        self.patcher = patch.multiple(main, STORE=self.store, JOBS=TrainingJobs(Path(self.temp.name)), DATA_DIR=Path(self.temp.name))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_project_selection_populates_training_dataset_with_choices(self):
        project = self.store.create_project("Narrator")
        dataset = self.store.project_dir(project["id"]) / "datasets" / "saved-dataset"
        dataset.mkdir()
        (dataset / "metadata.csv").write_text("", encoding="utf-8")
        training_update = self.ui.select_project(project["id"])[5]
        self.assertEqual(training_update["value"], "saved-dataset")
        self.assertEqual(training_update["choices"], [("saved-dataset", "saved-dataset")])

    def test_empty_project_clears_the_recording_picker(self):
        update = self.ui.select_project(None)[3]
        self.assertEqual(update["choices"], [])
        self.assertIsNone(update["value"])

    def test_recording_picker_uses_readable_labels_and_keeps_sample_ids(self):
        project = self.store.create_project("Narrator")
        prompt_id = self.store.import_prompts(project["id"], "text.txt", "A short prompt.", "lines")[0]
        sample_id = self.store.add_sample(project["id"], prompt_id, "recordings/raw/sample.wav", 2.5, "review", {"warnings": ["quiet"]})
        label, value = self.ui._sample_choices(project["id"])[0]
        self.assertIn("Review · 2.5s · A short prompt.", label)
        self.assertIn("check quality", label)
        self.assertEqual(value, sample_id)

    def test_empty_and_invalid_project_prompt_callbacks_do_not_raise(self):
        for project_id in (None, "arbitrary-id"):
            with self.subTest(project_id=project_id):
                for callback in (self.ui.resume_prompt, lambda pid: self.ui.current_prompt(pid, 0)):
                    result = callback(project_id)
                    self.assertEqual(result[0], 0)
                    self.assertEqual(result[2:], ("0 / 0", None))

    def test_saving_a_filtered_queue_keeps_the_full_project_queue(self):
        project = self.store.create_project("Narrator")
        self.store.import_prompts(project["id"], "text.txt", "First prompt.\nSecond prompt.", "lines")
        filtered = self.ui._queue_rows(project["id"], "First")
        result = self.ui.save_queue(project["id"], filtered, "First")
        self.assertIn("Clear", result[2])
        self.assertEqual(len(self.store.get_project(project["id"])["prompts"]), 2)

    def test_successful_recording_clears_the_take_before_the_next_prompt(self):
        project = self.store.create_project("Narrator")
        prompt_ids = self.store.import_prompts(project["id"], "text.txt", "A first prompt.\nA second prompt.", "lines")
        recording = Path(self.temp.name) / "take.wav"
        with wave.open(str(recording), "wb") as audio:
            audio.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
            audio.writeframes(b"".join(struct.pack("<h", int(4000 * math.sin(2 * math.pi * 220 * index / 22050))) for index in range(44100)))
        result = self.ui.record_sample(project["id"], 0, prompt_ids[0], str(recording), "accept")
        self.assertTrue(result[0].startswith("ACCEPTED"))
        self.assertEqual(result[4], 1)
        self.assertIsNone(result[6])

    def test_short_audio_durations_are_readable_in_dataset_choices(self):
        project = self.store.create_project("Short sample")
        dataset = self.store.project_dir(project["id"]) / "datasets" / "tiny"
        dataset.mkdir()
        (dataset / "metadata.csv").write_text("", encoding="utf-8")
        (dataset / "dataset.json").write_text('{"sample_count":1,"total_seconds":2}')
        self.assertIn("1 sample / 2 sec", self.ui._dataset_choices(project["id"])[0][0])

    def test_fresh_ui_has_six_ordered_steps_and_starts_with_project(self):
        async def build():
            return self.ui.build_app()
        app = asyncio.run(build())
        tabs = [component for component in app.config["components"] if component["type"] == "tabitem"]
        self.assertEqual([tab["props"]["label"] for tab in tabs], ["1 · Project", "2 · Text", "3 · Record", "4 · Dataset", "5 · Train", "6 · Voice"])
        self.assertEqual([tab["props"]["interactive"] for tab in tabs], [True, False, False, False, False, False])
        selected = next(component for component in app.config["components"] if component["type"] == "tabs")
        self.assertEqual(selected["props"]["selected"], 1)

    def test_forward_navigation_returns_inline_help_when_not_ready(self):
        update, message = self.ui.navigate_step(None, 1, 2)
        self.assertEqual(update["selected"], 1)
        self.assertIn("Step 1", message)

    def test_project_selection_clears_transient_view_and_restores_source(self):
        project = self.store.create_project("Narrator")
        self.store.import_prompts(project["id"], "text.txt", "My saved text.", "lines")
        result = self.ui.reset_project_view(project["id"])
        self.assertEqual(result[0], "My saved text.")
        self.assertIsNone(result[1])
        self.assertEqual(result[2], "lines")
        self.assertIsNone(result[3])

    def test_launch_allows_persistent_project_media_and_exports(self):
        async def launch():
            with patch.object(self.ui.gr.Blocks, "launch") as server:
                self.ui.launch_app()
                self.assertEqual(server.call_args.kwargs["allowed_paths"], [
                    str(Path(self.temp.name) / "projects"),
                    str(Path(self.temp.name) / "exports"),
                ])
        asyncio.run(launch())
