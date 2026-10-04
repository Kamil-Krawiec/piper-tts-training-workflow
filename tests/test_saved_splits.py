"""Saved dataset membership must reach Piper without another random split."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class SavedSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "splits").mkdir()
        (self.root / "audio").mkdir()
        self.ids = {"train": [], "validation": [], "test": []}
        self.index = []
        with (self.root / "metadata.csv").open("w") as stream:
            writer = csv.writer(stream, delimiter="|")
            for i in range(712):
                split = "train" if i < 605 else "validation" if i < 676 else "test"
                name = f"{i}.wav"
                text = f"Sentence {i}."
                self.ids[split].append(str(i))
                self.index.append({"sample_id": str(i), "filename": name, "text": text, "split": split})
                writer.writerow([name, text])
                (self.root / "audio" / name).touch()
        self.write_splits()

    def write_splits(self):
        (self.root / "sample-index.json").write_text(json.dumps(self.index))
        for name, ids in self.ids.items():
            (self.root / "splits" / f"{name}.json").write_text(json.dumps(ids))

    def test_saved_membership_is_complete_disjoint_and_preserves_csv_order(self):
        from app.datasets import saved_split_indices
        actual = saved_split_indices(self.root)
        self.assertEqual(actual, {"train": list(range(605)), "validation": list(range(605, 676)),
                                  "test": list(range(676, 712))})
        self.ids["validation"].append("0")
        self.write_splits()
        with self.assertRaises(ValueError):
            saved_split_indices(self.root)

    def test_supervisor_counts_all_605_training_rows_without_another_holdout(self):
        import os
        from unittest.mock import Mock, patch
        from app import train_run
        config = {"run_dir": str(self.root), "dataset_dir": str(self.root), "split_mode": "saved",
                  "csv_path": str(self.root / "metadata.csv"), "audio_dir": str(self.root / "audio"),
                  "cache_dir": str(self.root / "cache"), "config_path": str(self.root / "voice.json"),
                  "voice_name": "test", "sample_rate": 22050, "espeak_voice": "pl", "device": "cpu",
                  "training_mode": "scratch", "max_epochs": 250, "batch_size": 32,
                  "batch_size_mode": "manual", "num_workers": 1, "num_workers_mode": "manual",
                  "torch_threads": 2, "torch_threads_mode": "manual",
                  "validation_split": 0.1, "num_test_examples": 5}
        path = self.root / "run-config.json"
        path.write_text(json.dumps(config))
        child = Mock(pid=os.getpid())
        child.wait.return_value = 0
        with patch.object(train_run.subprocess, "Popen", return_value=child), \
             patch.object(train_run, "_sample_hardware"):
            self.assertEqual(train_run.run(path), 0)
        saved = json.loads(path.read_text())
        self.assertEqual(saved["training_samples"], 605)
        self.assertEqual(saved["split_counts"], {"train": 605, "validation": 71, "test": 36})
        self.assertEqual(saved["steps_per_epoch"], 19)
        command = saved["command"]
        self.assertEqual(command[command.index("--data.validation_split") + 1], "0.0")
        self.assertEqual(command[command.index("--data.num_test_examples") + 1], "0")
        self.assertEqual(command[command.index("--data.csv_path") + 1], str(self.root / "metadata.csv"))

    @unittest.skipUnless(importlib.util.find_spec("piper"), "Piper training runtime required")
    def test_piper_setup_uses_exact_saved_membership_even_after_seed_changes(self):
        from app.train_data import SavedSplitDataModule
        import torch
        for seed in (42, 99):
            torch.manual_seed(seed)
            data = SavedSplitDataModule(dataset_dir=str(self.root), csv_path=str(self.root / "metadata.csv"),
                audio_dir=str(self.root / "audio"), cache_dir=str(self.root / "cache"),
                config_path=str(self.root / "voice.json"), voice_name="test", espeak_voice="pl")
            data.setup("fit")
            self.assertEqual(data.train_dataset.indices, list(range(605)))
            self.assertEqual(data.val_dataset.indices, list(range(605, 676)))
            self.assertEqual(data.test_dataset.indices, list(range(676, 712)))
