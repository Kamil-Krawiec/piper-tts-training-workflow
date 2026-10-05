"""Local synthesis overrides use Piper's CLI and reject invalid values."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.inference import synthesize


class SynthesisTests(unittest.TestCase):
    def test_defaults_and_overrides_are_passed_before_literal_text(self):
        for overrides in ({}, {"noise_scale": 0, "noise_w": 0.5, "length_scale": 1.2}):
            with self.subTest(overrides=overrides), patch("app.inference.subprocess.run", return_value=Mock(returncode=0)) as run:
                synthesize(Path("voice.onnx"), Path("voice.onnx.json"), "--test text", Path("sample.wav"), **overrides)
                command = run.call_args.args[0]
                self.assertEqual(command[-2:], ["--", "--test text"])
                for name in ("noise_scale", "noise_w", "length_scale"):
                    flag = "--" + name.replace("_", "-")
                    if name in overrides:
                        self.assertEqual(command[command.index(flag) + 1], str(float(overrides[name])))
                    else:
                        self.assertNotIn(flag, command)

    def test_invalid_parameters_never_launch_piper(self):
        for name, values in {"noise_scale": [-1, float("nan"), "bad"],
                             "noise_w": [-1, float("inf")], "length_scale": [0, -1, float("inf")]}.items():
            for value in values:
                with self.subTest(name=name, value=value), patch("app.inference.subprocess.run") as run:
                    with self.assertRaisesRegex(ValueError, name):
                        synthesize(Path("voice.onnx"), Path("voice.onnx.json"), "Text", Path("out.wav"), **{name: value})
                    run.assert_not_called()
