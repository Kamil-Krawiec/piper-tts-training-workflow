"""Refresh the pinned browser pitch processor; run with Python's standard library."""

import hashlib
from io import BytesIO
from pathlib import Path
import tarfile
from urllib.request import urlopen

VERSION = "2.1.1"
SHA256 = "360c233284a44c30f6a7a20578e48f2011c988c86b737d422bc73da37bf8a24b"
URL = f"https://registry.npmjs.org/@soundtouchjs/audio-worklet/-/audio-worklet-{VERSION}.tgz"


def main():
    with urlopen(URL, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("SoundTouch archive checksum mismatch")
    destination = Path(__file__).resolve().parent.parent / "app" / "static"
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as archive:
        for source, target in (("package/.dist/soundtouch-processor.js", "soundtouch-processor.js"),
                               ("package/LICENSE", "SOUNDTOUCH-LICENSE")):
            with archive.extractfile(source) as stream:
                (destination / target).write_bytes(stream.read())
    print(f"Vendored SoundTouch {VERSION}; archive checksum verified.")


if __name__ == "__main__":
    main()
