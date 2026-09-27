import math
import tempfile
import unittest
import wave
from pathlib import Path

from app.audio import inspect_wav
from app.export import validate_voice_name


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

    def test_voice_names_reject_path_separators_and_spaces(self):
        self.assertEqual(validate_voice_name("pl_PL-kamil-medium"), "pl_PL-kamil-medium")
        for name in ("../voice", "voice name", "voice.onnx"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_voice_name(name)


if __name__ == "__main__":
    unittest.main()
