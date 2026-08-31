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
    serve = commands.add_parser(
        "serve", help="run the demand-driven restore sidecar (replaces a recompute warmer)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8431)
    serve.add_argument("--swap", default="http://127.0.0.1:9069")
    serve.add_argument("--manifest-root", default="~/.cfrproxy/cache")
    serve.add_argument("--store-root", default="~/.kvrosetta/admitted")
    serve.add_argument("--no-capture", action="store_true",
                       help="do not save warm slots automatically")
    serve.add_argument("--capture-min-tokens", type=int, default=4096)
    serve.add_argument("--capture-interval", type=float, default=20.0)
    pre = commands.add_parser(
        "prewarm", help="give a model its own attachment for a prefix, before it is needed")
    pre.add_argument("--model", required=True, help="llama-swap model name")
    pre.add_argument("--manifest", required=True, help="a cfrproxy prefix manifest")
    pre.add_argument("--swap", default="http://127.0.0.1:9069")
    pre.add_argument("--store-root", default="~/.kvrosetta/admitted")
    pre.add_argument("--from-scratch", action="store_true",
                     help="ignore any prior attachment and prefill the whole prefix")
    pre.add_argument("--allow-wake", action="store_true",
                     help="load the model if parked; off by default so this cannot become "
                          "the recompute warmer it replaces")
    args = parser.parse_args(argv)

    if args.command == "prewarm":
        from kv_rosetta.prewarm import prewarm_cli
        return prewarm_cli(args)

    if args.command == "serve":
        from kv_rosetta.daemon.server import SidecarConfig, build_server

        sidecar = build_server(SidecarConfig(
            host=args.host, port=args.port, swap=args.swap,
            manifest_root=args.manifest_root, store_root=args.store_root))
        # Deliberately no warm loop. A sidecar that reaches out to keep caches hot
        # recreates the behaviour it replaces: it wakes parked models and pays a full
        # prefill on a schedule whether or not anyone asked. This serves restores on
        # demand, and `models_woken` staying zero is the property that says so.
        print(f"kv-rosetta sidecar on http://{args.host}:{sidecar.port}  "
              f"store={args.store_root}  swap={args.swap}", flush=True)
        if not args.no_capture:
            # On by default. A cache is only missed once it is gone, so a capture that waits
            # to be asked for never happens. This sends no requests and wakes nothing: it
            # reads slot status and saves state that already exists on loaded models.
            import threading

            from kv_rosetta.daemon.capture import CaptureLoop

            def restore_on_load(model: str, slot: int):
                """The largest published prefix that already has an artifact for this model.

                Largest first because the whole point is the prompts that cost the most to
                prefill; a small prefix restored into the slot would occupy it and block the
                big one. Returns None when nothing matches, which the loop logs and skips --
                a model with no attachment simply prefills natively, as it does today.
                """
                try:
                    prefixes = sidecar.known_prefixes()
                except Exception as exc:
                    print(f"[capture] cannot read prefixes: {str(exc)[:120]}", flush=True)
                    return None
                # Ordered by what each artifact actually covers, not by the manifest's
                # est_tokens: that field counts the whole request rather than the
                # cacheable prefix, and ranking on it picked a 9,146-token attachment
                # over a 32,624-token one for the same model.
                # Relevance first, size second. Ranking on size alone restored a
                # 74,607-token attachment captured from a different harness over the
                # 32,624-token one this model's own traffic produced; it matched nothing
                # and the request prefilled cold anyway.
                from kv_rosetta.daemon.capture import same_model
                candidates = []
                for entry in prefixes:
                    fingerprint = str(entry.get("fingerprint", ""))
                    found = sidecar.find_artifact(fingerprint, model)
                    if found is None:
                        continue
                    covered = int((found.manifest or {}).get("prompt_token_count") or 0)
                    own = same_model(str(entry.get("model", "")), model)
                    candidates.append((own, covered, fingerprint))
                for own, covered, fingerprint in sorted(candidates, reverse=True):
                    result = sidecar.ensure(fingerprint, model, slot)
                    with sidecar._lock:
                        sidecar.stats.restores_served += 1
                    return {"prefix": fingerprint[:12], "covers_tokens": covered, **result}
                return None

            loop = CaptureLoop(args.swap, args.store_root,
                               min_tokens=args.capture_min_tokens,
                               interval=args.capture_interval,
                               restorer=restore_on_load,
                               log=lambda m: print(f"[capture] {m}", flush=True))
            threading.Thread(target=loop.run_forever, daemon=True,
                             name="kvx-capture").start()
        try:
            sidecar.serve_forever()
        except KeyboardInterrupt:
            sidecar.shutdown()
        return 0

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
