"""Assemble and verify the trained decoder bundled with the anonymous artifact."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / "decoder_manifest.json").read_text(encoding="utf-8"))
    folder = HERE / "checkpoints"
    folder.mkdir(exist_ok=True)
    archive = folder / "decoder.zip.part"
    print("Assembling the bundled trained decoder (about 516 MB).", flush=True)
    checksum = hashlib.sha256()
    total = 0
    with archive.open("wb") as stream:
        for part in manifest["parts"]:
            source = (HERE / part["path"]).resolve()
            if not source.is_relative_to(HERE / "weights"):
                raise ValueError("Invalid weight part path")
            if not source.is_file():
                raise FileNotFoundError(f"Missing {part['path']}; download and extract the full repository ZIP.")
            block = source.read_bytes()
            if len(block) != part["bytes"] or hashlib.sha256(block).hexdigest() != part["sha256"]:
                raise ValueError("Decoder part checksum mismatch: " + part["path"])
            checksum.update(block)
            total += len(block)
            stream.write(block)
    if total != manifest["archive_bytes"] or checksum.hexdigest() != manifest["archive_sha256"]:
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
