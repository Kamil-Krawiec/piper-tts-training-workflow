"""The supervisor persists resolved settings without needing a real GPU."""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app import train_run


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_resolved_cpu_settings_are_saved_before_training(self):
        csv_path = self.root / "metadata.csv"
        csv_path.write_text("a.wav|a\n" * 30, encoding="utf-8")
        config_path = self.root / "run-config.json"
        config_path.write_text(json.dumps({"run_dir": str(self.root), "csv_path": str(csv_path),
                                           "device": "cpu", "training_mode": "scratch", "max_epochs": 2000,
                                           "batch_size_mode": "auto", "num_workers_mode": "auto",
                                           "torch_threads_mode": "auto", "validation_split": 0.1,
                                           "num_test_examples": 5}), encoding="utf-8")
        child = Mock(pid=os.getpid())
        child.wait.return_value = 0
        with patch.object(train_run, "build_training_command", return_value=["trainer"]), \
             patch.object(train_run, "effective_cpu_count", return_value=8), \
             patch.object(train_run, "physical_cpu_count", return_value=8), \
             patch.object(train_run, "_sample_hardware"), \
             patch.object(train_run.subprocess, "Popen", return_value=child):
            self.assertEqual(train_run.run(config_path), 0)
        saved = json.loads(config_path.read_text())
        self.assertEqual(saved["batch_size"], 4)
        self.assertEqual(saved["num_workers"], 1)
        self.assertEqual(saved["torch_threads"], 7)
        self.assertEqual(saved["steps_per_epoch"], 6)
        self.assertEqual(saved["total_optimizer_steps"], 24000)
        self.assertEqual(saved["command"], ["trainer"])

    def test_child_environment_keeps_callbacks_importable_from_run_directory(self):
        environment = train_run._environment({"run_dir": str(self.root), "torch_threads": 2,
                                              "torch_interop_threads": 1})
        self.assertEqual(environment["PYTHONPATH"].split(os.pathsep)[0], str(Path(train_run.__file__).resolve().parent.parent))

    def test_telemetry_is_machine_readable(self):
        path = self.root / "hardware-metrics.jsonl"
        stop = threading.Event()
        with patch.object(train_run, "cpu_snapshot", return_value=({"total_percent": 25}, {})), \
             patch.object(train_run, "gpu_snapshot", return_value={"compute_percent": 90}), \
             patch.object(stop, "wait", side_effect=lambda seconds: stop.set()):
            train_run._sample_hardware(os.getpid(), path, stop)
        self.assertEqual(json.loads(path.read_text().splitlines()[0])["gpu"]["compute_percent"], 90)

    def test_cuda_probe_records_oom_and_keeps_memory_headroom(self):
        peaks = {4: 1, 8: 3, 16: 8, 32: 20}
        def trial(command, **kwargs):
            size = int(Path(kwargs["cwd"]).name)
            if size == 64:
                kwargs["stdout"].write(b"CUDA out of memory")
                return Mock(returncode=1)
            (kwargs["cwd"] / "probe-result.json").write_text(json.dumps({"peak_vram_bytes": peaks[size] * 1024**3}))
            return Mock(returncode=0)
        config = {"run_dir": str(self.root), "num_workers": 4, "gpu_batch_candidates": [4, 8, 16, 32, 64],
                  "gpu_memory_margin": 0.85}
        with patch.object(train_run, "build_training_command", return_value=["trainer"]), \
             patch.object(train_run, "gpu_snapshot", return_value={"name": "test GPU", "memory_total_bytes": 24 * 1024**3}), \
             patch.object(train_run.subprocess, "run", side_effect=trial):
            size, outcomes = train_run._probe(config, {}, train_count=100)
        self.assertEqual(size, 32)
        self.assertEqual(outcomes[-1]["outcome"], "oom")
        self.assertEqual(config["gpu"], "test GPU")


if __name__ == "__main__":
    unittest.main()
