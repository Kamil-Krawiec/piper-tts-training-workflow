"""Exercise scheduling against Piper's actual manual optimization loop."""
import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("piper") and importlib.util.find_spec("lightning"), "Piper training runtime required")
class LearningScheduleTests(unittest.TestCase):
    def test_manual_optimizers_decay_once_per_epoch_and_log_used_rates(self):
        import torch
        from torch.utils.data import DataLoader
        from lightning.pytorch import LightningModule, Trainer
        from lightning.pytorch.loggers import CSVLogger
        from piper.train.vits.lightning import VitsModel
        from app.training_metrics import LearningRateSchedule
        import csv

        class Probe(VitsModel):
            def __init__(self):
                LightningModule.__init__(self)
                self.automatic_optimization = False
                self.batch_size = 1
                self.save_hyperparameters({"learning_rate": 0.1, "learning_rate_d": 0.2,
                    "betas": (0.8, 0.99), "betas_d": (0.5, 0.9), "eps": 1e-9,
                    "lr_decay": 0.5, "lr_decay_d": 0.25})
                self.model_g = torch.nn.Linear(1, 1)
                self.model_d = torch.nn.Linear(1, 1)

            def _compute_loss(self, batch):
                return self.model_g(batch).square().mean(), self.model_d(batch).square().mean()

            def train_dataloader(self):
                return DataLoader(torch.ones(2, 1), batch_size=1)

        with tempfile.TemporaryDirectory() as temporary:
            trainer = Trainer(max_epochs=3, accelerator="cpu", logger=CSVLogger(temporary),
                enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                callbacks=[LearningRateSchedule()])
            trainer.fit(Probe())
            self.assertAlmostEqual(trainer.optimizers[0].param_groups[0]["lr"], 0.0125)
            self.assertAlmostEqual(trainer.optimizers[1].param_groups[0]["lr"], 0.003125)
            self.assertEqual([s.scheduler.last_epoch for s in trainer.lr_scheduler_configs], [3, 3])
            with (Path(trainer.logger.log_dir) / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            used = [(float(row["lr_g"]), float(row["lr_d"])) for row in rows if row.get("lr_g")]
            self.assertEqual(len(used), 3)
            for actual, expected in zip(used, [(0.1, 0.2), (0.05, 0.05), (0.025, 0.0125)]):
                for value, target in zip(actual, expected):
                    self.assertAlmostEqual(value, target)
