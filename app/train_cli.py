"""Run Piper training with the narrow checkpoint allowlist it needs."""

from pathlib import PosixPath
import inspect
import json
import logging
import os
import sys
from pathlib import Path

import torch

if os.environ.get("PIPER_TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["PIPER_TORCH_THREADS"]))
if os.environ.get("PIPER_TORCH_INTEROP_THREADS"):
    torch.set_num_interop_threads(int(os.environ["PIPER_TORCH_INTEROP_THREADS"]))


def main() -> None:
    # PyTorch 2.6 rejects PosixPath in otherwise weights-only Piper checkpoints.
    # Keep weights_only=True and allow only this harmless metadata type.
    from piper.train import __main__ as piper_cli

    base_cli = piper_cli.VitsLightningCLI
    model_parameters = inspect.signature(piper_cli.VitsModel.__init__).parameters
    logger = logging.getLogger(__name__)

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
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, dict):
                raise ValueError("Fine-tuning checkpoint is missing its model state dictionary")
            self._initial_state_dict = state_dict
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

        def before_fit(self) -> None:
            """Initialize from checkpoint weights without resuming its old training epoch."""
            fit_config = self.config[self.subcommand]
            if fit_config.get("ckpt_path"):
                self.model.load_state_dict(self._initial_state_dict)
                del self._initial_state_dict
                # LightningCLI builds Trainer.fit kwargs from config_init after this hook.
                self.config_init[self.subcommand]["ckpt_path"] = None
                logger.info("Loaded fine-tuning weights; starting a new training run at epoch 0.")
            run_config_path = os.environ.get("PIPER_RUN_CONFIG_PATH")
            if run_config_path:
                path = Path(run_config_path)
                config = json.loads(path.read_text(encoding="utf-8"))
                hparams = self.model.hparams
                epochs = int(config["max_epochs"])
                config["effective_lr_schedule"] = {
                    "generator_initial_lr": float(hparams.learning_rate),
                    "discriminator_initial_lr": float(hparams.learning_rate_d),
                    "generator_per_epoch_decay": float(hparams.lr_decay),
                    "discriminator_per_epoch_decay": float(hparams.lr_decay_d),
                    "generator_final_ratio": float(hparams.lr_decay) ** epochs,
                    "discriminator_final_ratio": float(hparams.lr_decay_d) ** epochs,
                    "max_epochs": epochs,
                }
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
                os.replace(temporary, path)

    piper_cli.VitsLightningCLI = CheckpointCompatibleCLI

    with torch.serialization.safe_globals([PosixPath]):
        piper_cli.main()


if __name__ == "__main__":
    main()
