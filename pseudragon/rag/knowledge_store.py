"""Serialized, recoverable changes to active institutional knowledge."""

from functools import wraps
from pathlib import Path
from threading import RLock
import hashlib
import json
import os
import tempfile

KNOWLEDGE_LOCK = RLock()


def revision(path):
    file = Path(path)
    return hashlib.sha256(file.read_bytes() if file.exists() else b"").hexdigest()


def atomic_json(path, data):
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=file.parent, prefix=".knowledge-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, file)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def current_store(method):
    """Serialize updates and reload active data before using a retained manager."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with KNOWLEDGE_LOCK:
            file = Path(self.preference_file)
            self.preferences = (
                json.loads(file.read_text(encoding="utf-8"))
                if file.exists()
                else self._load_preferences()
            )
            self.preferences.setdefault("pii_classification", {})
            token = revision(self.preference_file)
            if getattr(self, "_loaded_revision", None) != token:
                self._column_embeddings.clear()
            self._loaded_revision = token
            return method(self, *args, **kwargs)

    return wrapped
