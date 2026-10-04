"""Use Piper's preprocessing and loaders with our dataset's saved membership."""
from pathlib import Path
import logging

from piper.train.vits.dataset import VitsDataModule
from torch.utils.data import Subset

from app.datasets import saved_split_indices


class SavedSplitDataModule(VitsDataModule):
    def __init__(self, dataset_dir: str | None = None, **kwargs):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        if self.dataset_dir:
            kwargs["validation_split"] = 0.0
            kwargs["num_test_examples"] = 0
        super().__init__(**kwargs)

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.dataset_dir is None:
            return
        indices = saved_split_indices(self.dataset_dir)
        full_dataset = self.train_dataset.dataset
        if len(full_dataset) != sum(map(len, indices.values())):
            raise ValueError("Piper skipped dataset audio; saved split membership cannot be preserved")
        self.train_dataset = Subset(full_dataset, indices["train"])
        self.val_dataset = Subset(full_dataset, indices["validation"])
        self.test_dataset = Subset(full_dataset, indices["test"])
        logging.getLogger(__name__).info("Using saved dataset splits: %s train / %s validation / %s test",
            len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
