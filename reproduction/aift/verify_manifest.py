"""Verify the bytes of every distributed file using only the standard library."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for entry in manifest["files"]:
        path = HERE / entry["path"]
        assert path.resolve().is_relative_to(HERE)
        data = path.read_bytes()
        assert len(data) == entry["bytes"], entry["path"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
    print(json.dumps({"status": "PASS", "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
