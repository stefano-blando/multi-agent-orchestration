"""Lightweight persistence for strategic memory across process restarts."""

from __future__ import annotations

import json
import os
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState


def snapshot_path() -> str:
    raw = os.getenv("STATE_SNAPSHOT_PATH", "state_snapshot.json").strip()
    if os.path.isabs(raw):
        return raw
    return os.path.join(os.getcwd(), raw)


def load_snapshot(state: "GameState") -> bool:
    path = snapshot_path()
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    apply = getattr(state, "apply_snapshot", None)
    if callable(apply):
        apply(data)
        return True
    return False


def save_snapshot(state: "GameState") -> bool:
    path = snapshot_path()
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    to_snapshot = getattr(state, "to_snapshot", None)
    if not callable(to_snapshot):
        return False
    payload = to_snapshot()
    if not isinstance(payload, dict):
        return False
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".snapshot-", suffix=".json", dir=folder, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except OSError:
        return False
    return True
