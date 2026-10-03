import tempfile
import unittest
import wave
from pathlib import Path

from app.bundles import export_bundle, import_bundle
from app.datasets import create_dataset, write_metadata
from app.training import build_training_command, validate_training_config


def write_wav(path: Path):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(22050)
        audio.writeframes(b"\x00\x00" * 22050)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.samples = []
        for index, duration in enumerate((8, 10, 12, 14, 16, 18, 20, 22)):
            audio = self.root / f"{index:03}.wav"
            write_wav(audio)
            self.samples.append({
                "id": f"id-{index}", "text": f"Przykładowe zdanie numer {index}.",
                "audio_file": str(audio), "duration_seconds": duration,
            })

    def tearDown(self):
        self.temp.cleanup()

    def test_dataset_subsets_are_deterministic_and_duration_targeted(self):
        small = create_dataset(self.samples, self.root / "small", 60, seed=42)
        large = create_dataset(self.samples, self.root / "large", 120, seed=42)
        self.assertLessEqual(set(small["sample_ids"]), set(large["sample_ids"]))
        self.assertEqual(small["evaluation_sample_ids"], large["evaluation_sample_ids"])
        self.assertEqual(small["sample_ids"], create_dataset(self.samples, self.root / "again", 60, seed=42)["sample_ids"])
        self.assertEqual(small["seed"], 42)

    def test_metadata_uses_pipe_delimiter_and_relative_audio_names(self):
        source = self.root / "one.wav"
        write_wav(source)
        sample = {"id": "1", "text": "Piper działa.", "audio_file": str(source)}
        path = write_metadata([sample], self.root / "dataset")
        self.assertEqual(path.read_text().strip(), "000001.wav|Piper działa.")
        self.assertTrue((self.root / "dataset" / "audio" / "000001.wav").exists())

    def test_bundle_round_trip_and_path_traversal_rejection(self):
        dataset = self.root / "dataset"
        (dataset / "audio").mkdir(parents=True)
        write_wav(dataset / "audio" / "one.wav")
        (dataset / "metadata.csv").write_text("one.wav|Tekst\n")
        (dataset / "dataset.json").write_text('{"schema_version":1,"sample_count":1,"total_seconds":1}')
        (dataset / "sample-index.json").write_text('[{"sample_id":"sample-1","filename":"one.wav","text":"Tekst","split":"train"}]')
        (dataset / "project-info.json").write_text('{"language":"pl_PL","espeak_voice":"pl","sample_rate":22050}')
        (dataset / "splits").mkdir()
        (dataset / "splits" / "train.json").write_text('["sample-1"]')
        (dataset / "splits" / "validation.json").write_text('[]')
        (dataset / "splits" / "test.json").write_text('[]')
        archive = export_bundle(dataset, self.root / "export.zip", {"language": "pl_PL"})
        imported = import_bundle(archive, self.root / "imported")
        self.assertEqual((imported / "audio" / "one.wav").stat().st_size, 44144)
        (dataset / "sample-index.json").write_text('[{"sample_id":"sample-1","filename":"../outside.wav","text":"Tekst","split":"train"}]')
        invalid_index = export_bundle(dataset, self.root / "invalid-index.zip", {"language": "pl_PL"})
        with self.assertRaisesRegex(ValueError, "must match metadata"):
            import_bundle(invalid_index, self.root / "invalid-index")

        import zipfile
        malicious = self.root / "bad.zip"
        with zipfile.ZipFile(malicious, "w") as bundle:
            bundle.writestr("../escape.txt", "bad")
        with self.assertRaises(ValueError):
            import_bundle(malicious, self.root / "bad-import")


class TrainingCommandTests(unittest.TestCase):
    def test_checkpoint_interval_and_rates_are_validated_and_forwarded(self):
        import json
        base = {"voice_name": "voice", "csv_path": "/data/metadata.csv", "audio_dir": "/data/audio",
                "sample_rate": 22050, "espeak_voice": "pl", "cache_dir": "/data/cache",
                "config_path": "/data/config.json", "device": "cpu", "batch_size": 4,
                "training_mode": "scratch", "checkpoint_interval": 7, "learning_rate": 0.0001,
                "learning_rate_d": 0.00005}
        command = build_training_command(base)
        self.assertEqual(command[command.index("--model.learning_rate") + 1], "0.0001")
        callbacks = json.loads(command[command.index("--trainer.callbacks") + 1])
        intervals = [c["init_args"]["every_n_epochs"] for c in callbacks
                     if c["class_path"].endswith("ModelCheckpoint") and c["init_args"].get("save_top_k") == -1]
        self.assertEqual(intervals, [7])
        self.assertTrue(any(c.get("init_args", {}).get("monitor") == "val_loss" for c in callbacks))
        for field, invalid in [("checkpoint_interval", 0), ("checkpoint_interval", 2.5),
                               ("learning_rate", 0), ("learning_rate_d", float("nan"))]:
            with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                build_training_command({**base, field: invalid})

    def test_missing_finetune_checkpoint_returns_validation_help(self):
        errors = validate_training_config({"training_mode": "finetune", "checkpoint": None}, cuda_available=False)
        self.assertIn("Fine-tuning requires an existing checkpoint file.", errors)

    def test_finetuning_uses_checkpoint_but_scratch_uses_only_optional_warmstart(self):
        base = {
            "voice_name": "pl_PL-kamil-medium", "csv_path": "/data/metadata.csv",
            "audio_dir": "/data/audio", "sample_rate": 22050, "espeak_voice": "pl",
            "cache_dir": "/data/cache", "config_path": "/data/config.json",
            "device": "cpu", "batch_size": 8,
        }
        finetune = build_training_command({**base, "training_mode": "finetune", "checkpoint": "/ckpt/base.ckpt"})
        scratch = build_training_command({**base, "training_mode": "scratch", "vocoder_warmstart_checkpoint": "/ckpt/warm.ckpt"})
        self.assertIn("--ckpt_path", finetune)
        self.assertNotIn("--weights_only", finetune)
        self.assertNotIn("--ckpt_path", scratch)
        self.assertNotIn("--weights_only", scratch)
        self.assertIn("--model.vocoder_warmstart_ckpt", scratch)
        self.assertIn("--trainer.accelerator", finetune)
        self.assertIn("--trainer.max_epochs", finetune)
        self.assertIn("--trainer.logger", finetune)
        self.assertIn("lightning.pytorch.loggers.CSVLogger", " ".join(finetune))

    def test_explicit_cuda_is_invalid_when_container_has_no_cuda(self):
        config = {
            "dataset_dir": "/missing", "sample_rate": 22050, "espeak_voice": "pl",
            "training_mode": "scratch", "device": "cuda", "voice_name": "voice",
            "csv_path": "/missing/metadata.csv", "audio_dir": "/missing/audio",
            "cache_dir": "/cache", "config_path": "/run/config.json", "batch_size": 4,
        }
        errors = validate_training_config(config, cuda_available=False)
        self.assertTrue(any("CUDA was selected" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
