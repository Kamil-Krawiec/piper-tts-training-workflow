"""Run Piper training with the narrow checkpoint allowlist it needs."""

from pathlib import PosixPath
import inspect
import sys
from pathlib import Path

import torch


def main() -> None:
    # PyTorch 2.6 rejects PosixPath in otherwise weights-only Piper checkpoints.
    # Keep weights_only=True and allow only this harmless metadata type.
    from piper.train import __main__ as piper_cli

    base_cli = piper_cli.VitsLightningCLI
    model_parameters = inspect.signature(piper_cli.VitsModel.__init__).parameters

    class CheckpointCompatibleCLI(base_cli):
        def _parse_ckpt_path(self) -> None:
            """Load current model hparams, ignoring obsolete fields saved by old Piper."""
            if not self.config.get("subcommand"):
                return
            ckpt_path = self.config[self.config.subcommand].get("ckpt_path")
            if not ckpt_path or not Path(ckpt_path).is_file():
                return

            # Keep Lightning's safe weights-only load; the enclosing safe_globals
            # context only adds PosixPath, used by the curated legacy checkpoint.
            checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
            hparams = checkpoint.get("hyper_parameters", {})
            hparams.pop("_instantiator", None)
            hparams = {
                name: value
                for name, value in hparams.items()
                if name in model_parameters or name == "_class_path"
            }
            if not hparams:
                return
            if "_class_path" in hparams:
                hparams = {
                    "class_path": hparams.pop("_class_path"),
                    "dict_kwargs": hparams,
                }
            config = {self.config.subcommand: {"model": hparams}}
            try:
                self.config = self.parser.parse_object(config, self.config)
            except SystemExit:
                sys.stderr.write("Parsing of ckpt_path hyperparameters failed!\n")
                raise

    piper_cli.VitsLightningCLI = CheckpointCompatibleCLI

    with torch.serialization.safe_globals([PosixPath]):
        piper_cli.main()


if __name__ == "__main__":
    main()
