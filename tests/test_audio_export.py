import io
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from app.audio import inspect_wav
from app.export import validate_voice_name
from app.checkpoints import inference_config


class AudioAndExportTests(unittest.TestCase):
    def test_audio_checks_return_duration_and_clipping_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            frames = [int(32767 * math.sin(index * 0.1)) for index in range(22050 * 2)]
            frames[100] = 32767
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(22050)
                audio.writeframes(b"".join(frame.to_bytes(2, "little", signed=True) for frame in frames))
            result = inspect_wav(path)
        self.assertEqual(result["duration_seconds"], 2.0)
        self.assertIn("Audio contains clipped peaks.", result["warnings"])

    def test_darkman_config_is_fetched_once_and_explicit_missing_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            checkpoint = cache / "pl_PL-darkman-medium/checkpoint.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"weights")
            config_bytes = b'{"audio":{"sample_rate":22050},"espeak":{"voice":"pl"},"phoneme_id_map":{"a":[1]}}'
            with patch("app.checkpoints.urllib.request.urlopen", return_value=io.BytesIO(config_bytes)) as download:
                config = inference_config(checkpoint, None, cache)
                self.assertEqual(json.loads(config.read_text()), json.loads(config_bytes))
                self.assertEqual(inference_config(checkpoint, None, cache), config)
                download.assert_called_once()
                self.assertIn("config.json", download.call_args.args[0].full_url)
                with self.assertRaisesRegex(ValueError, "config"):
                    inference_config(checkpoint, str(cache / "missing.json"), cache)

    def test_voice_names_reject_path_separators_and_spaces(self):
        self.assertEqual(validate_voice_name("pl_PL-kamil-medium"), "pl_PL-kamil-medium")
        for name in ("../voice", "voice name", "voice.onnx"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_voice_name(name)


if __name__ == "__main__":
    unittest.main()
