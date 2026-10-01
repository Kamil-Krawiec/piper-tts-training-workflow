"""Persisted training jobs and progress shown after a browser reconnect."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.training import TrainingJobs, _process_exists


class TrainingJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project_id = "project-1"

    def test_run_uses_one_directory_for_process_config_and_metrics(self):
        run_id = "run-1"
        run_dir = self.root / "projects" / self.project_id / "runs" / run_id
        config = {"run_dir": str(run_dir), "cache_dir": str(run_dir / "cache"), "config_path": str(run_dir / "voice.onnx.json"), "max_epochs": 10}
        jobs = TrainingJobs(self.root)
        fake_process = Mock(pid=os.getpid())
        with patch("app.training.build_training_command", return_value=[sys.executable, "-c", "pass"]), patch("app.training.subprocess.Popen", return_value=fake_process) as launch:
            self.assertEqual(jobs.start(self.project_id, config, run_id=run_id), run_id)
        self.assertEqual(launch.call_args.args[0][1:3], ["-m", "app.train_run"])
        saved = json.loads((run_dir / "run-config.json").read_text())
        self.assertEqual(saved["config_path"], str(run_dir / "voice.onnx.json"))
        self.assertEqual(saved["cache_dir"], str(run_dir / "cache"))

    def test_running_job_survives_manager_recreation(self):
        config = {"max_epochs": 2}
        jobs = TrainingJobs(self.root)
        with patch("app.training.build_training_command", return_value=[sys.executable, "-c", "pass"]), patch("app.training.subprocess.Popen", return_value=Mock(pid=os.getpid())):
            run_id = jobs.start(self.project_id, config)
        reopened = TrainingJobs(self.root)
        self.assertEqual(reopened.status(self.project_id)["status"], "training")

    def test_metrics_report_completed_epochs_and_loss_series(self):
        run_dir = self.root / "projects" / self.project_id / "runs" / "run-1"
        metrics = run_dir / "metrics" / "version_0" / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("epoch,step,loss_g,val_loss\n0,10,3.5,\n0,11,,2.5\n1,20,2.7,\n", encoding="utf-8")
        (run_dir / "run-config.json").write_text(json.dumps({"max_epochs": 10}))
        (run_dir / "status.json").write_text(json.dumps({"status": "training", "pid": os.getpid()}))
        status = TrainingJobs(self.root).status(self.project_id)
        self.assertEqual(status["progress"]["current_epoch"], 2)
        self.assertEqual(status["progress"]["completed_epochs"], 1)
        self.assertEqual(status["progress"]["max_epochs"], 10)
        self.assertEqual(len(status["progress"]["losses"]), 3)

    def test_permission_denied_still_means_process_exists(self):
        with patch("app.training.os.kill", side_effect=PermissionError):
            self.assertTrue(_process_exists(123))


if __name__ == "__main__":
    unittest.main()
