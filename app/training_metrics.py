"""Small Lightning callbacks for durable training timing and probe results."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from app.performance import estimate_runtime


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class RunModelCheckpoint(ModelCheckpoint):
    """Distinct roles must stay checkpointable even when their intervals match."""

    def __init__(self, role: str, **kwargs):
        self.role = role
        super().__init__(**kwargs)

    @property
    def state_key(self) -> str:
        return f"{super().state_key}:{self.role}"


class LearningRateSchedule(Callback):
    """Pinned Piper uses manual optimization, so Lightning never steps its schedulers."""

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        for label, optimizer in zip(("lr_g", "lr_d"), trainer.optimizers):
            pl_module.log(label, optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not pl_module.automatic_optimization:
            for config in trainer.lr_scheduler_configs:
                config.scheduler.step()


class TrainingMetrics(Callback):
    """Lightning callback using batch timings; warm-up batches never enter the ETA."""

    def __init__(self, path: str, steps_per_epoch: int, max_epochs: int, hourly_rate: float | None = None):
        self.path = Path(path)
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.hourly_rate = hourly_rate
        self.batch_times: list[float] = []
        self.started = 0.0
        self.batch_started = 0.0
        self.last_batch_finished = 0.0
        self.epoch_started = 0.0
        self.epoch_seconds: list[float] = []

    def on_train_start(self, trainer, pl_module) -> None:
        self.started = time.time()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_started = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self.batch_started = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.strategy.root_device.type == "cuda":
            import torch
            torch.cuda.synchronize()
        finished = time.monotonic()
        self.batch_times.append(finished - (self.last_batch_finished or self.batch_started))
        self.last_batch_finished = finished
        if len(self.batch_times) % 5 == 0:
            self._save(trainer)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self.epoch_seconds.append(time.time() - self.epoch_started)
        self._save(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        trainer.save_checkpoint(str(self.path.parent / "checkpoints" / "final.ckpt"))
        self._save(trainer)

    def _save(self, trainer) -> None:
        batches = len(self.batch_times)
        total = self.steps_per_epoch * self.max_epochs
        elapsed = max(0, time.time() - self.started)
        estimate = estimate_runtime(self.batch_times, batches, total, self.hourly_rate, elapsed_seconds=elapsed)
        _atomic_json(self.path, {"elapsed_seconds": elapsed,
                                 "current_epoch": min(self.max_epochs, trainer.current_epoch + 1),
                                 "global_step": trainer.global_step,
                                 "learning_rates": [optimizer.param_groups[0]["lr"] for optimizer in trainer.optimizers],
                                 "completed_batches": batches,
                                 "steps_per_epoch": self.steps_per_epoch,
                                 "total_optimizer_steps": total * 2,
                                 "average_epoch_seconds": sum(self.epoch_seconds[-5:]) / len(self.epoch_seconds[-5:]) if self.epoch_seconds else None,
                                 "steps_per_second": 1 / estimate["seconds_per_batch"] if estimate else None,
                                 "samples_per_second": trainer.datamodule.batch_size / estimate["seconds_per_batch"] if estimate else None,
                                 "estimate": estimate})


class ProbeMetrics(Callback):
    def __init__(self, path: str):
        self.path = Path(path)
        self.peak_device_used_bytes = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        import torch
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        self.peak_device_used_bytes = max(self.peak_device_used_bytes, total - free)
        _atomic_json(self.path, {"peak_vram_bytes": torch.cuda.max_memory_reserved(),
                                 "peak_device_used_bytes": self.peak_device_used_bytes,
                                 "completed_batches": batch_idx + 1})
