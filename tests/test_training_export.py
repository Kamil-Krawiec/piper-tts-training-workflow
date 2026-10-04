"""Durable run downloads preserve metrics and exclude temporary probe files."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from app.export import package_training_run


class TrainingArchiveTests(unittest.TestCase):
    def test_archive_keeps_metrics_and_checkpoints_but_excludes_probe_and_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            files = {'metrics/version_0/metrics.csv': 'epoch,val_mel\n57,0.388\n',
                     'checkpoints/best.ckpt': 'weights', 'run-config.json': '{}',
                     'pid': '123', 'probe/checkpoints/temporary.ckpt': 'probe'}
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            archive = package_training_run(root, Path(tmp) / 'training.zip')
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), {'metrics/version_0/metrics.csv', 'checkpoints/best.ckpt', 'run-config.json'})
                self.assertEqual(bundle.read('metrics/version_0/metrics.csv').decode(), files['metrics/version_0/metrics.csv'])

    def test_failed_archive_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            root.mkdir()
            (root / 'metrics.csv').write_text('metrics')
            archive = Path(tmp) / 'training.zip'
            with patch.object(zipfile.ZipFile, 'write', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    package_training_run(root, archive)
            self.assertFalse(archive.exists())
