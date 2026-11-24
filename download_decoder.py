"""Download and verify the trained decoder release asset."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / "decoder_manifest.json").read_text(encoding="utf-8"))
    folder = HERE / "checkpoints"
    folder.mkdir(exist_ok=True)
    archive = folder / "decoder.zip.part"
    print("Downloading the trained decoder (about 516 MB).", flush=True)
    with urllib.request.urlopen(manifest["url"], timeout=120) as response, archive.open("wb") as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: response.read(1024 * 1024), b""):
            checksum.update(block)
            stream.write(block)
    if checksum.hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Decoder archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != {"decoder/" + name for name in manifest["files"]}:
            raise ValueError("Unexpected checkpoint archive contents")
        for name, expected in manifest["files"].items():
            content = bundle.read("decoder/" + name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Decoder file checksum mismatch: " + name)
            target = folder / "decoder" / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(content)
    archive.unlink()
    print("Verified decoder saved to checkpoints/decoder")


if __name__ == "__main__":
    main()
