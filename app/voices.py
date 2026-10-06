"""Discover and import self-contained Piper ONNX/config pairs."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import uuid
import zipfile

from app.bundles import _safe_member

MAX_VOICE_BYTES = 1024 ** 3


def voice_choices(root: Path) -> list[tuple[str, str]]:
    choices = []
    root = root.resolve()
    for config in sorted(root.rglob("*.onnx.json")):
        model = config.with_name(config.name[:-5])
        if not all(path.is_file() and path.stat().st_size and path.resolve().is_relative_to(root)
                   and not path.is_symlink() for path in (model, config)):
            continue
        try:
            metadata = json.loads(config.read_text(encoding="utf-8"))
            language = metadata.get("espeak", {}).get("voice", "")
        except (ValueError, OSError, AttributeError):
            continue
        imported = model.parent.name.startswith("imported-")
        label = model.stem + (f" · Imported · {model.parent.name[-8:]}" if imported else "")
        choices.append((label + (f" · {language}" if language else ""), str(model.resolve())))
    return choices


def selected_voice(root: Path, model_path: str) -> Path:
    if not model_path or model_path not in {value for _, value in voice_choices(root)}:
        raise ValueError("Choose a saved voice from this project's list")
    return Path(model_path).resolve()


def validate_config(config: Path) -> dict:
    try:
        if config.stat().st_size > 1024 ** 2:
            raise ValueError("Configuration exceeds 1 MiB")
        data = json.loads(config.read_text(encoding="utf-8"))
        rate = data["audio"]["sample_rate"]
        if not isinstance(rate, int) or isinstance(rate, bool) or not 8000 <= rate <= 96000:
            raise ValueError("Unsupported sample rate")
        if not isinstance(data["phoneme_id_map"], dict) or not data["phoneme_id_map"]:
            raise ValueError("Missing phoneme map")
        if not isinstance(data["espeak"]["voice"], str) or not data["espeak"]["voice"]:
            raise ValueError("Missing eSpeak voice")
        return data
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Piper JSON configuration: {error}") from error


def validate_model(model: Path) -> None:
    from piper.config import PiperConfig

    try:
        PiperConfig.from_dict(validate_config(model.with_name(model.name + ".json")))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Piper configuration: {error}") from error

    import onnx
    import onnxruntime

    proto = onnx.load(model, load_external_data=False)

    def check_external_data(message):
        if isinstance(message, onnx.TensorProto) and (message.external_data or message.data_location == onnx.TensorProto.EXTERNAL):
            raise ValueError("Import a self-contained ONNX model; external tensor files are unsupported")
        # Include tensors in subgraphs, sparse initializers, and local functions.
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                for child in value if field.is_repeated else (value,):
                    check_external_data(child)

    check_external_data(proto)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    if not {"input", "input_lengths", "scales"}.issubset({item.name for item in session.get_inputs()}):
        raise ValueError("ONNX model is not a Piper voice")


def import_voice(uploads: list[Path], destination: Path) -> Path:
    files = [Path(path) for path in uploads or []]
    if not files or len(files) > 2 or sum(path.stat().st_size for path in files) > MAX_VOICE_BYTES:
        raise ValueError("Upload one voice ZIP or an ONNX + matching JSON pair, up to 1 GiB")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".voice-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        if len(files) == 1 and files[0].suffix.lower() == ".zip":
            with zipfile.ZipFile(files[0]) as bundle:
                members = bundle.infolist()
                if len(members) > 16 or sum(member.file_size for member in members) > MAX_VOICE_BYTES:
                    raise ValueError("Voice ZIP exceeds the 16-entry / 1 GiB limit")
                names = set()
                for member in members:
                    _safe_member(member.filename)
                    if (member.filename in names
                            or stat.S_ISLNK(member.external_attr >> 16)):
                        raise ValueError("Unsafe or duplicate voice ZIP entry")
                    names.add(member.filename)
                pairs = [member for member in members if member.filename.endswith(".onnx")]
                if len(pairs) != 1 or pairs[0].filename + ".json" not in names:
                    raise ValueError("Voice ZIP must contain exactly one ONNX + matching JSON pair")
                for name in (pairs[0].filename, pairs[0].filename + ".json"):
                    with bundle.open(name) as source, (staging / PurePosixPath(name).name).open("wb") as target:
                        shutil.copyfileobj(source, target)
        else:
            if len(files) != 2 or len({path.name for path in files}) != 2:
                raise ValueError("Upload both the ONNX file and its matching .onnx.json")
            for path in files:
                shutil.copyfile(path, staging / path.name)
        models = list(staging.glob("*.onnx"))
        if len(models) != 1:
            raise ValueError("Upload exactly one .onnx model")
        model = models[0]
        config = model.with_name(model.name + ".json")
        if not config.is_file() or not model.stat().st_size:
            raise ValueError("Missing matching .onnx.json or empty model")
        validate_config(config)
        validate_model(model)
        final = destination / f"imported-{uuid.uuid4().hex}"
        staging.rename(final)
        return final / model.name
