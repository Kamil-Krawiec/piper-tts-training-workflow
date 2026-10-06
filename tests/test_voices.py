"""Voice import and discovery contracts; no Piper installation needed."""

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch
import zipfile

from app.voices import import_voice, voice_choices, selected_voice, validate_model


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        self.model = self.root / "voice.onnx"
        self.model.write_bytes(b"model")
        self.config = self.root / "voice.onnx.json"
        self.config.write_text(json.dumps({"audio": {"sample_rate": 22050}, "espeak": {"voice": "pl"},
                                           "phoneme_id_map": {"a": [1]}}))

    def test_pair_import_and_duplicate_import_preserve_both_models(self):
        with patch("app.voices.validate_model"):
            first = import_voice([self.model, self.config], self.models)
            second = import_voice([self.config, self.model], self.models)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), b"model")
        self.assertEqual(len(voice_choices(self.models)), 2)
        self.assertEqual(selected_voice(self.models, str(first)), first.resolve())
        self.assertIn("Imported", voice_choices(self.models)[0][0])

    def test_voice_is_not_discoverable_until_validation_completes(self):
        def validate(staged):
            self.assertTrue(staged.is_file())
            self.assertEqual(voice_choices(self.models), [])
        with patch("app.voices.validate_model", side_effect=validate):
            import_voice([self.model, self.config], self.models)
        self.assertEqual(len(voice_choices(self.models)), 1)

    @unittest.skipUnless(importlib.util.find_spec("piper"), "Piper is not installed")
    def test_upstream_piper_config_is_checked_before_loading_onnx(self):
        with self.assertRaisesRegex(ValueError, "num_symbols"):
            validate_model(self.model)
        config = json.loads(self.config.read_text())
        config.update(num_symbols=256, num_speakers=1, phoneme_type="invalid")
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "PhonemeType"):
            validate_model(self.model)

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX is not installed")
    def test_external_tensor_paths_are_rejected_before_runtime_load(self):
        import onnx
        config = json.loads(self.config.read_text())
        config.update(num_symbols=256, num_speakers=1)
        self.config.write_text(json.dumps(config))
        tensor = onnx.TensorProto(name="weights", dims=[1], data_type=onnx.TensorProto.FLOAT,
                                  data_location=onnx.TensorProto.EXTERNAL)
        tensor.external_data.add(key="location", value="../private")
        graph = onnx.helper.make_graph([], "external", [], [], [tensor])
        self.model.write_bytes(onnx.helper.make_model(graph).SerializeToString())
        with self.assertRaisesRegex(ValueError, "external tensor"):
            validate_model(self.model)

    def test_invalid_or_ambiguous_archive_leaves_no_import(self):
        for extra in ("../escape", "nested/other.onnx", "voice.onnx"):
            archive = self.root / "voice.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.write(self.model, self.model.name)
                    bundle.write(self.config, self.config.name)
                    bundle.writestr(extra, b"other")
            with self.subTest(extra=extra), patch("app.voices.validate_model"), self.assertRaises(ValueError):
                import_voice([archive], self.models)
            self.assertFalse(self.models.exists() and list(self.models.iterdir()))

    def test_bad_config_and_model_failure_are_atomic(self):
        for content in ("{}", "not json", '{"audio":{"sample_rate":0}}'):
            self.config.write_text(content)
            with self.assertRaises(ValueError):
                import_voice([self.model, self.config], self.models)
        self.config.write_text(json.dumps({"audio": {"sample_rate": 22050}, "espeak": {"voice": "pl"}, "phoneme_id_map": {"a": [1]}}))
        with patch("app.voices.validate_model", side_effect=ValueError("Invalid ONNX")), self.assertRaises(ValueError):
            import_voice([self.model, self.config], self.models)
        self.assertEqual(list(self.models.iterdir()), [])

    def test_discovery_excludes_missing_empty_and_external_models(self):
        self.models.mkdir()
        (self.models / "empty.onnx").touch()
        (self.models / "empty.onnx.json").write_text("{}")
        (self.models / "outside.onnx").symlink_to(self.model)
        (self.models / "outside.onnx.json").symlink_to(self.config)
        self.assertEqual(voice_choices(self.models), [])
        with self.assertRaises(ValueError):
            selected_voice(self.models, str(self.model))

    def test_archive_limits_and_symlinks_are_rejected(self):
        archive = self.root / "voice.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            member = zipfile.ZipInfo("link.onnx")
            member.create_system = 3
            member.external_attr = 0o120777 << 16
            bundle.writestr(member, "target")
        with self.assertRaises(ValueError):
            import_voice([archive], self.models)
        with patch("app.voices.MAX_VOICE_BYTES", 2), self.assertRaises(ValueError):
            import_voice([self.model, self.config], self.models)
