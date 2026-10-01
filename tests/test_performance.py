"""Contracts for automatic training settings and measured estimates."""

import unittest

from app.performance import cpu_batch_for, cpu_threads_for, epoch_cap_for, estimate_runtime, select_batch_size, workers_for
from app.training import build_training_command


class PerformanceTests(unittest.TestCase):
    def test_auto_batch_respects_peak_memory_margin(self):
        trials = [(8, "pass", 5), (16, "pass", 9), (32, "pass", 19), (64, "oom", None)]
        self.assertEqual(select_batch_size(trials, total_vram=24, margin=0.85), 32)
        self.assertEqual(select_batch_size(trials, total_vram=20, margin=0.85), 16)

    def test_auto_batch_fails_when_no_candidate_passes(self):
        with self.assertRaisesRegex(ValueError, "No batch size"):
            select_batch_size([(8, "oom", None)], total_vram=24)

    def test_auto_cpu_settings_reserve_compute_for_training(self):
        self.assertEqual(workers_for("cuda", physical_cores=8, logical_cores=16), 6)
        self.assertEqual(workers_for("cuda", physical_cores=8, logical_cores=16, training_samples=40), 1)
        self.assertEqual(workers_for("cuda", physical_cores=8, logical_cores=16, training_samples=120), 2)
        self.assertEqual(workers_for("cpu", physical_cores=8, logical_cores=16), 2)
        self.assertEqual(cpu_threads_for("cpu", physical_cores=8, logical_cores=16, workers=2), 6)
        self.assertEqual(cpu_threads_for("cuda", physical_cores=8, logical_cores=16, workers=6), 2)

    def test_auto_cpu_batch_uses_ram_and_longest_utterance(self):
        gib = 1024 ** 3
        self.assertEqual(cpu_batch_for(200, 32 * gib, 6), 16)
        self.assertEqual(cpu_batch_for(200, 16 * gib, 10), 8)
        self.assertEqual(cpu_batch_for(200, 8 * gib, 10), 4)
        self.assertEqual(cpu_batch_for(200, 32 * gib, 25), 4)
        self.assertEqual(cpu_batch_for(6, 32 * gib, 6), 4)

    def test_epoch_caps(self):
        self.assertEqual(epoch_cap_for("finetune"), 1000)
        self.assertEqual(epoch_cap_for("scratch"), 2000)

    def test_estimate_requires_real_steps_and_ignores_warmup(self):
        self.assertIsNone(estimate_runtime([3, 2], completed_batches=4, total_batches=100, hourly_rate=None))
        estimate = estimate_runtime([5] * 5 + [0.4] * 20, completed_batches=25, total_batches=100, hourly_rate=0.34)
        self.assertAlmostEqual(estimate["seconds_per_batch"], 0.4)
        self.assertAlmostEqual(estimate["remaining_seconds"], 30)
        self.assertAlmostEqual(estimate["remaining_cost"], 30 / 3600 * 0.34)
        with_elapsed = estimate_runtime([5] * 5 + [0.4] * 20, 25, 100, 0.34, elapsed_seconds=100)
        self.assertAlmostEqual(with_elapsed["total_cost"], 130 / 3600 * 0.34)

    def test_command_uses_workers_and_one_device(self):
        base = {"voice_name": "voice", "csv_path": "/data/metadata.csv", "audio_dir": "/data/audio",
                "sample_rate": 22050, "espeak_voice": "pl", "cache_dir": "/data/cache",
                "config_path": "/data/config.json", "training_mode": "scratch", "batch_size": 16,
                "num_workers": 4, "device": "cuda"}
        command = build_training_command(base)
        self.assertEqual(command[command.index("--data.num_workers") + 1], "4")
        self.assertEqual(command[command.index("--trainer.devices") + 1], "1")
        self.assertEqual(command[command.index("--trainer.accelerator") + 1], "gpu")
        cpu = build_training_command({**base, "device": "cpu"})
        self.assertEqual(cpu[cpu.index("--trainer.accelerator") + 1], "cpu")


if __name__ == "__main__":
    unittest.main()
