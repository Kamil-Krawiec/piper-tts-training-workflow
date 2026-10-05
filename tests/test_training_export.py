"""Durable run downloads preserve metrics and exclude temporary probe files."""
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path, PosixPath
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


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ExportCompatibilityTests(unittest.TestCase):
    def test_export_allows_legacy_path_metadata_with_weights_only_loading(self):
        import torch
        from app.export_cli import main
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.ckpt"
            torch.save({"hyper_parameters": {"dataset": PosixPath("/data/dataset")}}, checkpoint)
            def upstream_export():
                data = torch.load(checkpoint, weights_only=True)
                self.assertEqual(data["hyper_parameters"]["dataset"], PosixPath("/data/dataset"))
            with patch("piper.train.export_onnx.main", side_effect=upstream_export) as export:
                main()
                export.assert_called_once()
            self.assertNotIn(PosixPath, torch.serialization.get_safe_globals())
