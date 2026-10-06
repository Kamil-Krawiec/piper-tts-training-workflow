"""Metrics downloads exclude training artifacts and work during a run."""
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path, PosixPath
from unittest.mock import patch

from app.export import package_training_metrics


class TrainingArchiveTests(unittest.TestCase):
    def test_archive_contains_only_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            files = {'metrics/version_0/metrics.csv': 'epoch,val_mel\n57,0.388\n',
                     'checkpoints/best.ckpt': 'weights', 'run-config.json': '{}',
                     'training-metrics.json': '{}', 'hardware-metrics.jsonl': '{}\n', 'dataset/audio.wav': 'audio', 'train.log': 'log', 'pid': '123', 'probe/checkpoints/temporary.ckpt': 'probe'}
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            archive = package_training_metrics(root, Path(tmp) / 'training.zip')
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), {'metrics/version_0/metrics.csv', 'training-metrics.json', 'hardware-metrics.jsonl'})
                self.assertEqual(bundle.read('metrics/version_0/metrics.csv').decode(), files['metrics/version_0/metrics.csv'])

    def test_active_metrics_snapshot_ignores_partial_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            csv = root / 'metrics/version_0/metrics.csv'
            csv.parent.mkdir(parents=True)
            csv.write_bytes(b'epoch,loss\n1,0.5\n2,')
            hardware = root / 'hardware-metrics.jsonl'
            hardware.write_bytes(b'{"cpu": 12}\n{"cpu":')
            archive = package_training_metrics(root, Path(tmp) / 'metrics.zip')
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(bundle.read('metrics/version_0/metrics.csv'), b'epoch,loss\n1,0.5\n')
                self.assertEqual(bundle.read('hardware-metrics.jsonl'), b'{"cpu": 12}\n')
            csv.write_bytes(b'epoch,loss\n1,0.5\n2,0.4\n')
            package_training_metrics(root, archive)
            with zipfile.ZipFile(archive) as bundle:
                self.assertIn(b'2,0.4\n', bundle.read('metrics/version_0/metrics.csv'))

    def test_no_metrics_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'No metrics'):
                package_training_metrics(Path(tmp), Path(tmp) / 'metrics.zip')

    def test_failed_archive_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            root.mkdir()
            (root / 'training-metrics.json').write_text('{}')
            archive = Path(tmp) / 'training.zip'
            with patch.object(zipfile.ZipFile, 'writestr', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    package_training_metrics(root, archive)
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
