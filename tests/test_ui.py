"""UI callback contracts; run with the app's Gradio environment."""

import asyncio
import json
import importlib.util
import math
import os
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
        self.environment = patch.dict(os.environ, {"PIPER_DATA_DIR": self.temp.name,
                                                   "GRADIO_TEMP_DIR": str(Path(self.temp.name) / "tmp")})
        self.environment.start()
        self.addCleanup(self.environment.stop)
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
        timer_ids = {item["id"] for item in app.config["components"] if item["type"] == "timer"}
        picker_ids = {item["id"] for item in app.config["components"] if item["props"].get("label") in {"Trained voice run", "Saved checkpoint"}}
        self.assertTrue(any(picker_ids.issubset(set(event["outputs"])) for event in app.config["dependencies"]
                            if any(target[0] in timer_ids for target in event["targets"])))

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

    def test_training_start_uses_the_run_id_used_in_saved_config_paths(self):
        project = self.store.create_project("Narrator")
        dataset = self.store.project_dir(project["id"]) / "datasets" / "saved"
        dataset.mkdir(parents=True)
        with patch.object(self.ui, "validate_training_config", return_value=[]), patch.object(self.ui.JOBS, "start", return_value="run-1") as start, patch.object(self.ui, "_run_status_text", return_value="status"):
            self.ui.start_training(project["id"], "saved", "scratch", "", "", "cpu", 4, 42, 10)
        config = start.call_args.args[1]
        self.assertEqual(start.call_args.kwargs["run_id"], Path(config["run_dir"]).name)

    def test_training_view_explains_missing_loss_metrics(self):
        project = self.store.create_project("Narrator")
        state = {"run_id": "failed-run", "status": "failed", "progress": {
            "current_epoch": 0, "completed_epochs": 0, "max_epochs": 10, "losses": [],
        }}
        with patch.object(self.ui.JOBS, "status", return_value=state):
            summary, chart, _ = self.ui.training_view(project["id"])
        self.assertIn("No loss metrics were saved", summary)
        self.assertFalse(chart["visible"])

    def test_training_view_shows_saved_epoch_progress_and_loss_plot(self):
        project = self.store.create_project("Narrator")
        state = {"run_id": "run-1234", "status": "training", "log_tail": "latest log", "progress": {
            "current_epoch": 3, "completed_epochs": 2, "max_epochs": 10,
            "losses": [{"epoch": 1, "loss": 3.0, "series": "Training"}, {"epoch": 2, "loss": 2.5, "series": "Validation"}],
        }}
        with patch.object(self.ui.JOBS, "status", return_value=state):
            summary, chart, logs = self.ui.training_view(project["id"])
        self.assertIn("Epoch 3 of 10", summary)
        self.assertIn("20%", summary)
        self.assertTrue(chart["visible"])
        self.assertEqual(len(chart["value"]), 2)
        self.assertIn("latest log", logs)

    def test_voice_runs_only_include_saved_training_checkpoints(self):
        project = self.store.create_project("Narrator")
        root = self.store.project_dir(project["id"]) / "runs"
        for run_id in ("trained", "failed"):
            run = root / run_id
            run.mkdir(parents=True)
            (run / "run-config.json").write_text(json.dumps({"voice_name": "Narrator", "training_mode": "finetune", "max_epochs": 100}))
        checkpoint = root / "trained" / "metrics/version_0/checkpoints/epoch=99-step=200.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"trained")
        probe = root / "failed" / "probe/8/checkpoints/test.ckpt"
        probe.parent.mkdir(parents=True)
        probe.write_bytes(b"probe")
        choices = self.ui.run_choices(project["id"])
        self.assertEqual([value for _, value in choices], ["trained"])
        self.assertIn("Narrator", choices[0][0])
        self.assertIn("100", choices[0][0])
        self.assertIn("After epoch 100", self.ui.checkpoint_choices(project["id"], "trained")[0][0])

    def test_refresh_preserves_selected_run_and_checkpoint(self):
        project = self.store.create_project("Narrator")
        for run_id in ("older", "newer"):
            run = self.store.project_dir(project["id"]) / "runs" / run_id
            (run / "checkpoints").mkdir(parents=True)
            (run / "run-config.json").write_text('{}')
            (run / "checkpoints/final.ckpt").write_bytes(b"trained")
        checkpoint = str(self.store.project_dir(project["id"]) / "runs/older/checkpoints/final.ckpt")
        with patch.object(self.ui, "training_view", return_value=("summary", {}, "logs")):
            result = self.ui.refresh_runs(project["id"], "older", checkpoint)
        self.assertEqual(result[1]["value"], "older")
        self.assertEqual(result[4]["value"], checkpoint)
        self.assertTrue(result[5]["interactive"])

    def test_poll_does_not_rebuild_unchanged_dropdowns(self):
        project = self.store.create_project("Voice")
        with patch.object(self.ui, "training_view", return_value=("summary", {}, "logs")):
            first = self.ui.poll_training(project["id"], None, None, None)
            second = self.ui.poll_training(project["id"], None, None, first[-1])
        self.assertEqual(second[1], {"__type__": "update"})
        self.assertEqual(second[4], {"__type__": "update"})

    def test_export_defaults_to_latest_checkpoint_even_when_named_last(self):
        project = self.store.create_project("Narrator")
        run = self.store.project_dir(project["id"]) / "runs/trained"
        (run / "checkpoints/latest").mkdir(parents=True)
        older = run / "checkpoints/epoch-0000.ckpt"
        latest = run / "checkpoints/latest/last.ckpt"
        older.write_bytes(b"older")
        latest.write_bytes(b"latest")
        os.utime(older, (1, 1)); os.utime(latest, (2, 2))
        def export(checkpoint, config, output, name):
            self.assertEqual(checkpoint.name, latest.name)
            self.assertEqual(checkpoint.read_bytes(), b"latest")
            output.mkdir(parents=True)
            model = output / (name + '.onnx'); model.write_bytes(b'model')
            config = output / (name + '.onnx.json'); config.write_text('{}')
            return model, config
        with patch.object(self.ui, "export_onnx", side_effect=export):
            archive, message, _ = self.ui.export_run(project["id"], "trained", "Narrator")
        self.assertIsNotNone(archive, message)

    def test_checkpoint_download_copies_selected_weights_and_rejects_other_runs(self):
        project = self.store.create_project("Voice")
        run = self.store.project_dir(project["id"]) / "runs/trained"
        (run / "checkpoints/latest").mkdir(parents=True)
        checkpoint = run / "checkpoints/latest/last.ckpt"
        checkpoint.write_bytes(b"chosen weights")
        download, message = self.ui.download_run_checkpoint(project["id"], "trained", str(checkpoint))
        self.assertIsNotNone(download, message)
        self.assertEqual(Path(download).read_bytes(), b"chosen weights")
        checkpoint.write_bytes(b"new rolling weights")
        self.assertEqual(Path(download).read_bytes(), b"chosen weights")
        outside = Path(self.temp.name) / "unrelated.ckpt"
        outside.write_bytes(b"other run")
        rejected, message = self.ui.download_run_checkpoint(project["id"], "trained", str(outside))
        self.assertIsNone(rejected)
        self.assertIn("does not belong", message)

    def test_training_start_persists_custom_checkpoint_interval_and_learning_rates(self):
        project = self.store.create_project("Voice")
        dataset = self.store.project_dir(project["id"]) / "datasets/saved"
        dataset.mkdir(parents=True)
        with patch.object(self.ui, "validate_training_config", return_value=[]), \
             patch.object(self.ui.JOBS, "start", return_value="run-1") as start, \
             patch.object(self.ui, "_run_status_text", return_value="status"):
            self.ui.start_training(project["id"], "saved", "scratch", "", "", "cpu", 4, 42, 10,
                                   checkpoint_interval=7, learning_rate=0.0001, learning_rate_d=0.00005)
        config = start.call_args.args[1]
        self.assertEqual(config["checkpoint_interval"], 7)
        self.assertEqual(config["learning_rate"], 0.0001)
        self.assertEqual(config["learning_rate_d"], 0.00005)

    def test_imported_audio_summary_deduplicates_reimported_recordings(self):
        project = self.store.create_project("GPU voice")
        for name in ("imported-one", "imported-again"):
            dataset = self.store.project_dir(project["id"]) / "datasets" / name
            (dataset / "audio").mkdir(parents=True)
            (dataset / "metadata.csv").write_text("one.wav|Sentence\n")
            (dataset / "sample-index.json").write_text(json.dumps([
                {"sample_id": "same-recording", "filename": "one.wav", "text": "Sentence", "split": "test"}]))
            with wave.open(str(dataset / "audio/one.wav"), "wb") as audio:
                audio.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * 22050)
        summary = self.ui._project_summary(project["id"])
        self.assertIn("**1** imported recording", summary)
        self.assertIn("**1 sec** imported audio", summary)
        self.assertIn("**1** available recording", summary)

    def test_generate_sample_uses_selected_checkpoint_export(self):
        model = Path(self.temp.name) / "selected.onnx"
        with patch.object(self.ui, "export_run", return_value=("voice.zip", "Exported", {"value": str(model)})) as export, \
             patch.object(self.ui, "synthesize_model", return_value=("sample.wav", "Generated")) as synth:
            result = self.ui.generate_checkpoint_sample("project", "run", "Voice", "latest/last.ckpt", "Sentence")
        export.assert_called_once_with("project", "run", "Voice", "latest/last.ckpt")
        synth.assert_called_once_with(str(model), "Sentence")
        self.assertEqual(result[0], "sample.wav")
        self.assertEqual(result[2], "voice.zip")
        self.assertIn("run", result[1])

    def test_comparison_choices_include_saved_models_and_real_epoch_checkpoints(self):
        project = self.store.create_project("Narrator")
        root = self.store.project_dir(project["id"])
        model = root / "models/starting.onnx"
        model.parent.mkdir(exist_ok=True)
        model.write_bytes(b"model")
        model.with_name("starting.onnx.json").write_text('{}')
        for run_id, filename in (("earlier-run", "epoch=49-step=123.ckpt"), ("later-run", "best-epoch-0141.ckpt")):
            checkpoint = root / "runs" / run_id / "checkpoints" / filename
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            checkpoint.with_name("empty.ckpt").touch()
        sources = self.ui._comparison_sources(project["id"])
        self.assertEqual(len(sources), 3)
        self.assertEqual(sources[str(model)], ("Saved model · starting", None))
        self.assertTrue(any("After epoch 50" in label and run == "earlier-run" for label, run in sources.values()))
        self.assertTrue(any("Best validation · After epoch 142" in label and run == "later-run" for label, run in sources.values()))
        self.assertEqual(self.ui._comparison_sources(None), {})

    def test_comparison_supports_models_and_checkpoints_across_runs(self):
        sources = {"a.ckpt": ("Epoch 50", "run-a"), "b.ckpt": ("Epoch 142", "run-b"),
                   "voice.onnx": ("Saved model", None), "other.onnx": ("Other model", None)}
        for source_a, source_b in (("a.ckpt", "b.ckpt"), ("voice.onnx", "other.onnx"), ("voice.onnx", "b.ckpt")):
            with self.subTest(a=source_a, b=source_b), \
                 patch.object(self.ui, "_comparison_sources", return_value=sources), \
                 patch.object(self.ui, "generate_checkpoint_sample", return_value=("checkpoint.wav", "Generated", "voice.zip", {})) as generate, \
                 patch.object(self.ui, "synthesize_model", return_value=("model.wav", "Generated")) as synth:
                result = self.ui.compare_voices("project", "Voice", source_a, source_b, "Sentence")
                self.assertTrue(all(result[:2]))
                self.assertIn(sources[source_a][0], result[2])
                self.assertIn(sources[source_b][0], result[2])
                self.assertEqual(generate.call_args_list, [unittest.mock.call("project", sources[source][1], "Voice", source, "Sentence")
                                                          for source in (source_a, source_b) if sources[source][1]])
                self.assertEqual(synth.call_args_list, [unittest.mock.call(source, "Sentence")
                                                       for source in (source_a, source_b) if sources[source][1] is None])

    def test_comparison_validates_both_sources_and_clears_failed_audio(self):
        sources = {"a": ("A", "run"), "b": ("B", None)}
        with patch.object(self.ui, "_comparison_sources", return_value=sources), \
             patch.object(self.ui, "generate_checkpoint_sample", return_value=("a.wav", "OK", "a.zip", {})) as generate, \
             patch.object(self.ui, "synthesize_model", return_value=(None, "Failed")):
            for a, b, text in (("a", "a", "Text"), ("a", "outside-project", "Text"), ("a", "b", "  ")):
                self.assertEqual(self.ui.compare_voices("p", "V", a, b, text)[:2], (None, None))
            generate.assert_not_called()
            result = self.ui.compare_voices("p", "V", "a", "b", "Text")
            self.assertEqual(result, (None, None, "Comparison failed: Failed"))

    def test_comparison_refresh_preserves_selections_and_drops_missing_sources(self):
        with patch.object(self.ui, "_comparison_sources", return_value={"a": ("A", "run"), "b": ("B", None)}):
            a, b = self.ui.refresh_voice_comparison("project", "a", "b")
            self.assertEqual((a["value"], b["value"]), ("a", "b"))
            self.assertEqual(a["choices"], [("A", "a"), ("B", "b")])
            a, b = self.ui.refresh_voice_comparison("project", "deleted", "b")
            self.assertIsNone(a["value"])
            self.assertEqual(b["value"], "b")

    def test_empty_preview_text_does_not_export(self):
        with patch.object(self.ui, "export_run") as export:
            result = self.ui.generate_checkpoint_sample("project", "run", "Voice", "last.ckpt", "  ")
        export.assert_not_called()
        self.assertIsNone(result[0])
        self.assertIn("text", result[1].lower())

    def test_failed_export_does_not_generate_an_older_model(self):
        with patch.object(self.ui, "export_run", return_value=(None, "Export failed", {})), \
             patch.object(self.ui, "synthesize_model") as synth:
            result = self.ui.generate_checkpoint_sample("project", "run", "Voice", "last.ckpt", "Sentence")
        synth.assert_not_called()
        self.assertIsNone(result[0])
        self.assertIsNone(result[3]["value"])

    def test_rolling_exports_do_not_overwrite_each_other(self):
        project = self.store.create_project("Voice")
        run = self.store.project_dir(project["id"]) / "runs/trained"
        (run / "checkpoints/latest").mkdir(parents=True)
        checkpoint = run / "checkpoints/latest/last.ckpt"
        checkpoint.write_bytes(b"saved checkpoint")
        def export(snapshot, config, output, name):
            output.mkdir(parents=True)
            model = output / (name + ".onnx")
            model.write_bytes(snapshot.read_bytes())
            config = output / (name + ".onnx.json")
            config.write_text("{}")
            return model, config
        with patch.object(self.ui, "export_onnx", side_effect=export):
            first = self.ui.export_run(project["id"], "trained", "Voice", str(checkpoint))
            checkpoint.write_bytes(b"new checkpoint")
            second = self.ui.export_run(project["id"], "trained", "Voice", str(checkpoint))
        self.assertIsNotNone(first[0], first[1])
        self.assertIsNotNone(second[0], second[1])
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(Path(first[2]["value"]).read_bytes(), b"saved checkpoint")
        self.assertEqual(Path(second[2]["value"]).read_bytes(), b"new checkpoint")

    def test_reference_prompts_use_selected_runs_dataset(self):
        project = self.store.create_project("Voice")
        root = self.store.project_dir(project["id"])
        (root / "runs/selected").mkdir(parents=True)
        (root / "runs/selected/run-config.json").write_text(json.dumps({"dataset_id": "imported-test"}))
        (root / "datasets/imported-test").mkdir(parents=True)
        (root / "datasets/imported-test/sample-index.json").write_text(json.dumps([
            {"sample_id": "held-out", "text": "Reference sentence", "split": "test"}]))
        choices = self.ui.run_evaluation_choices(project["id"], "selected")
        self.assertEqual(choices[0][1], "held-out")
        self.assertEqual(self.ui.refresh_run_reference(project["id"], "selected", "held-out")["value"], "held-out")

    def test_failed_import_does_not_leave_a_dataset_for_project_metrics(self):
        project = self.store.create_project("Voice")
        def invalid_bundle(archive, destination):
            destination.mkdir(parents=True)
            (destination / "metadata.csv").write_text("one.wav|Sentence")
            raise ValueError("Invalid index")
        with patch.object(self.ui, "import_bundle", side_effect=invalid_bundle):
            result = self.ui.import_dataset_ui(project["id"], "bad.zip")
        self.assertIn("failed", result[2])
        self.assertEqual(self.ui._dataset_dirs(project["id"]), [])
