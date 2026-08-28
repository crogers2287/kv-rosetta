"""Command line entry point.

Manifest commands stay stdlib-only; container commands import numpy lazily so that
`validate` and `compat` keep working on a machine that only has the standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from kv_rosetta.manifest import ManifestError, compatibility, load


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _inspect(path: str) -> int:
    from kv_rosetta.container import ContainerError, read, verify

    try:
        artifact = read(path, mmap=True)
    except ContainerError as exc:
        print(f"kv-rosetta: {exc}", file=sys.stderr)
        return 2
    ok, reason = verify(path)
    blob = artifact.blob
    _json({
        "schema": artifact.header.get("schema"),
        "encoding": blob.get("encoding"),
        "opaque_format": blob.get("opaque_format"),
        "shape": blob.get("shape"),
        "offset": blob.get("offset"),
        "nbytes": blob.get("nbytes"),
        "kv": artifact.header.get("kv"),
        "producer": artifact.header.get("producer"),
        "identity": artifact.header.get("identity"),
        "integrity": {"ok": ok, "reason": reason},
    })
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kv-rosetta")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a KVX manifest")
    validate.add_argument("manifest")
    fingerprint = commands.add_parser("fingerprint", help="print model/cache ABI identity")
    fingerprint.add_argument("manifest")
    compat = commands.add_parser("compat", help="plan source-to-target cache transfer")
    compat.add_argument("source")
    compat.add_argument("target")
    inspect = commands.add_parser("inspect", help="summarise a .kvx artifact without loading tensors")
    inspect.add_argument("artifact")
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            data, abi = load(args.manifest)
            _json({
                "valid": True,
                "schema": data["schema"],
                "abi_fingerprint": abi.fingerprint(),
                "producer": data.get("producer"),
            })
        elif args.command == "fingerprint":
            _, abi = load(args.manifest)
            print(abi.fingerprint())
        elif args.command == "inspect":
            return _inspect(args.artifact)
        else:
            _, source = load(args.source)
            _, target = load(args.target)
            result = compatibility(source, target)
            result["source_abi"] = source.fingerprint()
            result["target_abi"] = target.fingerprint()
            _json(result)
        return 0
    except ManifestError as exc:
        print(f"kv-rosetta: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
