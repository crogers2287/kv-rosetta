from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


class StoreError(ValueError):
    pass


_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


class IdentityError(StoreError):
    """An externally supplied identity is not a well-formed digest."""


def require_digest(value: str, name: str = "fingerprint") -> str:
    """Validate an externally supplied hash before it is ever used in a path.

    A prefix manifest is written by another process. If its fingerprint reached the
    filesystem unchecked, a value like ``../../etc`` would escape the store root.
    Nothing that fails this check may be used to build a path.
    """
    if not isinstance(value, str) or not _HEX64.match(value):
        raise IdentityError(
            f"{name} must be 64 lowercase hex characters, got {value!r}"
        )
    return value


def fingerprint(
    provider: str,
    model: str,
    system_sha256: str,
    tools_sha256: str,
) -> str:
    payload = provider + "\x00" + model + "\x00" + system_sha256 + "\x00" + tools_sha256
    return hashlib.sha256(payload.encode()).hexdigest()


def model_key(provider: str, model: str) -> str:
    payload = provider + "\x00" + model
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def default_root() -> Path:
    env = os.environ.get("KVROSETTA_STORE")
    if env:
        return Path(env)
    return Path.home() / ".kvrosetta" / "store"


@dataclass(frozen=True)
class Record:
    fingerprint: str
    model_key: str
    provider: str
    model: str
    tier: int
    nbytes: int
    l0_sha256: str
    created: float
    last_used: float
    path: str


class Store:
    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            root = default_root()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        self._conn = sqlite3.connect(str(self.root / "index.sqlite"))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                fingerprint TEXT PRIMARY KEY,
                model_key TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tier INTEGER NOT NULL,
                nbytes INTEGER NOT NULL,
                l0_sha256 TEXT NOT NULL,
                created REAL NOT NULL,
                last_used REAL NOT NULL,
                path TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_artifacts_model_lastused "
            "ON artifacts(model_key, last_used)"
        )
        self._conn.commit()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None

    def _record(self, row: sqlite3.Row) -> Record:
        return Record(
            fingerprint=row[0],
            model_key=row[1],
            provider=row[2],
            model=row[3],
            tier=row[4],
            nbytes=row[5],
            l0_sha256=row[6],
            created=row[7],
            last_used=row[8],
            path=row[9],
        )

    def path_for(self, provider: str, model: str, fp: str) -> Path:
        """Resolve an artifact path, refusing anything that could leave the root.

        The full digest is used as the filename: truncating it for tidiness would
        trade a real collision risk for cosmetics. A short form is fine as a display
        label, never as an identity.
        """
        require_digest(fp)
        mk = model_key(provider, model)
        path = (self.root / mk / (fp + ".kvx")).resolve()
        root = self.root.resolve()
        if not (path == root or root in path.parents):
            raise IdentityError(f"resolved path {path} escapes store root {root}")
        return path

    def put(
        self,
        *,
        fingerprint: str,
        provider: str,
        model: str,
        tier: int,
        l0_sha256: str,
        source: Path,
        now: float | None = None,
    ) -> Record:
        src = Path(source)
        if not src.exists():
            raise StoreError(f"source does not exist: {src}")
        mk = model_key(provider, model)
        dest = self.root / mk / (fingerprint[:24] + ".kvx")
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
        nbytes = dest.stat().st_size
        ts = time.time() if now is None else now
        existing = self._conn.execute(
            "SELECT created FROM artifacts WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        created = existing[0] if existing is not None else ts
        self._conn.execute(
            "INSERT OR REPLACE INTO artifacts "
            "(fingerprint, model_key, provider, model, tier, nbytes, l0_sha256, created, last_used, path) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (fingerprint, mk, provider, model, tier, nbytes, l0_sha256, created, ts, str(dest)),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        return self._record(row)

    def get(self, fingerprint: str) -> Record | None:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        if row is None:
            return None
        rec = self._record(row)
        if not Path(rec.path).exists():
            self._conn.execute(
                "DELETE FROM artifacts WHERE fingerprint=?", (fingerprint,)
            )
            self._conn.commit()
            return None
        return rec

    def touch(self, fingerprint: str, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        self._conn.execute(
            "UPDATE artifacts SET last_used=? WHERE fingerprint=?", (ts, fingerprint)
        )
        self._conn.commit()

    def delete(self, fingerprint: str) -> bool:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        if row is None:
            return False
        rec = self._record(row)
        try:
            p = Path(rec.path)
            if p.exists():
                p.unlink()
        except OSError:
            pass
        self._conn.execute("DELETE FROM artifacts WHERE fingerprint=?", (fingerprint,))
        self._conn.commit()
        return True

    def list(
        self, model_key: str | None = None, limit: int = 0
    ) -> tuple[Record, ...]:
        if model_key is not None:
            rows = self._conn.execute(
                "SELECT * FROM artifacts WHERE model_key=? ORDER BY last_used DESC",
                (model_key,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM artifacts ORDER BY last_used DESC"
            ).fetchall()
        if limit > 0:
            rows = rows[:limit]
        return tuple(self._record(r) for r in rows)

    def total_bytes(self, model_key: str | None = None) -> int:
        if model_key is not None:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(nbytes), 0) FROM artifacts WHERE model_key=?",
                (model_key,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(nbytes), 0) FROM artifacts"
            ).fetchone()
        return int(row[0])

    def prune(
        self,
        *,
        max_bytes: int = 0,
        max_entries: int = 0,
        max_age_seconds: float = 0.0,
        now: float | None = None,
    ) -> tuple[Record, ...]:
        ts = time.time() if now is None else now
        records_desc = self.list()
        records_asc = list(reversed(records_desc))
        remaining = {rec.fingerprint for rec in records_asc}
        deleted: list[Record] = []

        def _remove(rec: Record) -> None:
            try:
                p = Path(rec.path)
                if p.exists():
                    p.unlink()
            except OSError:
                pass
            self._conn.execute(
                "DELETE FROM artifacts WHERE fingerprint=?", (rec.fingerprint,)
            )
            self._conn.commit()
            remaining.discard(rec.fingerprint)

        # 1. age
        if max_age_seconds > 0:
            cutoff = ts - max_age_seconds
            for rec in records_asc:
                if rec.fingerprint in remaining and rec.last_used < cutoff:
                    _remove(rec)
                    deleted.append(rec)

        # 2. entry count
        if max_entries > 0:
            for rec in records_asc:
                if len(remaining) <= max_entries:
                    break
                if rec.fingerprint in remaining:
                    _remove(rec)
                    deleted.append(rec)

        # 3. total bytes
        if max_bytes > 0:
            current = self.total_bytes()
            for rec in records_asc:
                if current <= max_bytes:
                    break
                if rec.fingerprint in remaining:
                    _remove(rec)
                    current -= rec.nbytes
                    deleted.append(rec)

        return tuple(deleted)
