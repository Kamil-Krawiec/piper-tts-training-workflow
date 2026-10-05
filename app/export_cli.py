"""Export Piper checkpoints with the same narrow allowlist as training."""

from pathlib import PosixPath


def main() -> None:
    import torch
    from piper.train.export_onnx import main as export

    # Legacy Piper checkpoints store paths in metadata; keep weights-only loading.
    with torch.serialization.safe_globals([PosixPath]):
        export()


if __name__ == "__main__":
    main()
