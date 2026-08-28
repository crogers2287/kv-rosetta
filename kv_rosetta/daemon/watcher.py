from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WatcherError(ValueError):
    pass


@dataclass(frozen=True)
class PrefixManifest:
    path: Path
    data: dict
    mtime: float

    @property
    def fingerprint(self) -> str:
        return self.data["fingerprint"]

    @property
    def provider(self) -> str:
        return self.data["provider"]

    @property
    def model(self) -> str:
        return self.data["model"]

    @property
    def est_tokens(self) -> int:
        last = self.data.get("last_prompt_tokens")
        if last:
            return int(last)
        system_bytes = self.data.get("system_bytes", 0) or 0
        tools = self.data.get("tools", "")
        return int((system_bytes + len(tools)) / 3.5)


def load_manifests(root: Path | str, provider: str = "", model: str = "") -> tuple[PrefixManifest, ...]:
    root_path = Path(root)
    if not root_path.exists():
        return ()

    manifests: list[PrefixManifest] = []
    for path in sorted(root_path.rglob("*.json")):
        if path.name == "_identity.json":
            continue
        try:
            data: Any = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("schema") != 1:
            continue
        if provider and data.get("provider") != provider:
            continue
        if model and data.get("model") != model:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        manifests.append(PrefixManifest(path=path, data=data, mtime=mtime))

    manifests.sort(key=lambda m: m.mtime, reverse=True)
    return tuple(manifests)


def changed_since(manifests: tuple[PrefixManifest, ...], since: float) -> tuple[PrefixManifest, ...]:
    return tuple(m for m in manifests if m.mtime > since)
