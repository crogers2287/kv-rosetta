"""Keep the CPU-resident part of a model file in the kernel's page cache.

Most of a GGUF is copied to the GPU once at load and never read again. The exception is a
tensor deliberately placed on the CPU -- for Qwen3.8-Flash-Next that is the 28.8 GB
per-layer embedding table -- which is read from the mapped file for every token. When
another model loads and the kernel needs memory, those pages are evicted, and the next
token walks to the NAS. Measured on this host: prefill fell from about 720 tok/s to about
40, and decode from 64 to 8.

posix_fadvise(WILLNEED) asks the kernel to pull a range into page cache without copying it
into this process, so this costs no resident memory of its own and needs no privileges.
It is advisory: under real memory pressure the kernel may still evict, which is the honest
difference between this and mlock, and mlock needs a raised RLIMIT_MEMLOCK.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


class KeepWarmError(ValueError):
    pass


def check_range(path: Path, offset: int, length: int) -> int:
    """Validate a byte range against the file, returning its size."""
    if not path.is_file():
        raise KeepWarmError(f"{path} is not a file")
    size = path.stat().st_size
    if offset < 0:
        raise KeepWarmError(f"offset {offset} is negative")
    if length <= 0:
        raise KeepWarmError(f"length {length} is not positive")
    if offset + length > size:
        raise KeepWarmError(
            f"range {offset}..{offset + length} runs past the end of {path} ({size} bytes)")
    return size


def touch_once(path: Path, offset: int, length: int) -> float:
    """Ask the kernel to make one range resident. Returns seconds taken."""
    check_range(path, offset, length)
    started = time.time()
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_WILLNEED)
    finally:
        os.close(fd)
    return time.time() - started


def run(path: Path, offset: int, length: int, interval: float, *, once: bool = False,
        log=print) -> None:
    if interval <= 0:
        raise KeepWarmError(f"interval {interval} is not positive")
    check_range(path, offset, length)
    log(f"keep-warm: {length / 1e9:.1f} GB of {path.name} every {interval:.0f}s")
    while True:
        try:
            took = touch_once(path, offset, length)
            log(f"touched {length / 1e9:.1f} GB in {took:.1f}s")
        except OSError as exc:                    # a transient NAS error is not fatal
            log(f"touch failed: {exc}")
        if once:
            return
        time.sleep(interval)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, type=Path)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    run(args.path, args.offset, args.length, args.interval, once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
