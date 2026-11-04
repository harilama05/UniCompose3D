"""Atomic, ID-addressed local storage. Credentials never enter project files."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

ID = re.compile(r"^[a-f0-9]{32}$")


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.lock = threading.RLock()
        for kind in ("assets", "sources", "jobs", "scenes", "cache", "exports", "compositions", "composition_jobs"):
            (self.root / kind).mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, ident: str):
        if kind not in {"assets", "sources", "jobs", "scenes", "exports", "compositions", "composition_jobs"} or not ID.fullmatch(ident):
            raise ValueError("Invalid asset ID.")
        return self.root / kind / ident

    def create(self, kind: str, metadata: dict | None = None):
        ident = uuid.uuid4().hex
        directory = self.path(kind, ident)
        directory.mkdir()
        self.write(directory / "meta.json", {"id": ident, "created_at": now(), **(metadata or {})})
        return ident, directory

    def write(self, path: Path, value: dict):
        with self.lock:
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)

    def read(self, kind: str, ident: str):
        with self.lock:
            return json.loads((self.path(kind, ident) / "meta.json").read_text(encoding="utf-8"))

    def update(self, kind: str, ident: str, **changes):
        with self.lock:
            value = self.read(kind, ident)
            value.update(changes)
            self.write(self.path(kind, ident) / "meta.json", value)
            return value

    def list(self, kind: str):
        result = []
        for path in (self.root / kind).glob("*/meta.json"):
            try:
                result.append(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                continue
        return sorted(result, key=lambda v: v.get("created_at", ""), reverse=True)

    def cache_get(self, key: str):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Invalid cache key.")
        path = self.root / "cache" / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def cache_put(self, key: str, value: dict):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Invalid cache key.")
        self.write(self.root / "cache" / f"{key}.json", value)
