"""Command line entry point.

Manifest commands stay stdlib-only; container commands import numpy lazily so that
`validate` and `compat` keep working on a machine that only has the standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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
                """The best attachment this model has, ranked by relevance then size.

                Enumerates the STORE, not the cfrproxy prefix corpus. An attachment built
                by capturing this model's own traffic is keyed by a fingerprint derived
                from its tokens, which by construction never appears among the proxy's
                manifests -- so ranking over the corpus made exactly those artifacts
                invisible, which is the bug that made every capture useless.

                Relevance: a prefix absent from the corpus was captured from this model's
                own slot, so it is by definition this model's traffic. One present in the
                corpus is only relevant if its label denotes this model.
                """
                from kv_rosetta.daemon.capture import rank_restore_candidates, same_model
                try:
                    labels = {str(e.get("fingerprint", "")): str(e.get("model", ""))
                              for e in sidecar.known_prefixes()}
                except Exception as exc:
                    print(f"[capture] cannot read prefixes: {str(exc)[:120]}", flush=True)
                    labels = {}
                try:
                    objects = list(sidecar.store().list_objects())
                except Exception as exc:
                    print(f"[capture] cannot read store: {str(exc)[:120]}", flush=True)
                    return None

                candidates = []
                for obj in objects:
                    man = obj.manifest or {}
                    if man.get("runtime_model") != model:
                        continue
                    fingerprint = str(man.get("prefix_fingerprint") or "")
                    if not fingerprint:
                        continue
                    covered = int(man.get("prompt_token_count") or 0)
                    own = (fingerprint not in labels) or same_model(labels[fingerprint], model)
                    # Admit time, which the payload's mtime records exactly: nothing
                    # rewrites an object after it is admitted.
                    try:
                        seen_at = obj.path.stat().st_mtime
                    except OSError:
                        seen_at = 0.0
                    candidates.append((own, seen_at, covered, fingerprint))

                # Ranked by recency, NOT by size. There is no request to be relevant to at
                # model-load time, so this is a guess, and the useful property of a guess is
                # that being wrong repairs it. Recency has that: a miss is prefilled by the
                # server, capture admits that prefix, and it is the newest thing here next
                # time. Size does not -- a 75,523-token attachment outranked every Hermes
                # prompt on this host indefinitely, so each load restored 75k tokens the
                # request shared no prefix with and then paid the full 31k prefill anyway.
                # A wrong restore costs more than no restore, so the tie-break that matters
                # is which prefix traffic actually used last.
                for own, seen_at, covered, fingerprint in rank_restore_candidates(candidates):
                    result = sidecar.ensure(fingerprint, model, slot)
                    with sidecar._lock:
                        sidecar.stats.restores_served += 1
                    return {"prefix": fingerprint[:12], "covers_tokens": covered,
                            "own": own, **result}
                return None

            def admit_capture(model: str, basename: str, saved: dict):
                """Turn a saved slot into an artifact restore-on-load can find.

                Without this a capture is bytes on disk that nothing looks up: artifacts
                are keyed by prefix fingerprint and runtime model, and a bare .state
                carries neither. The fingerprint comes from the captured tokens, so
                traffic that never produced a cfrproxy manifest still gains a reusable
                attachment after its first prefill.
                """
                from kv_rosetta.adapters import ggsq_envelope
                from kv_rosetta.adapters.admitted_path import AdmittedPath
                from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter
                from kv_rosetta.daemon.capture import prefix_fingerprint

                store_root = Path(args.store_root).expanduser()
                raw = (store_root / basename).read_bytes()
                token_ids = list(ggsq_envelope.decode_prompt_tokens(
                    ggsq_envelope.parse_file_envelope(raw).token_ids))
                fingerprint = prefix_fingerprint(token_ids)
                # Already held? Re-admitting mints a second copy of the same bytes AND
                # stamps it as the newest attachment for this model, which would hand the
                # load restore a stale prefix on the strength of a fresh timestamp -- the
                # ranking in `rank_restore_candidates` is only as honest as the times it
                # reads. A daemon restart re-sees every resident slot, so without this the
                # duplicate is minted on every restart.
                for existing in sidecar.store().list_objects():
                    prior = existing.manifest or {}
                    if (prior.get("runtime_model") == model
                            and prior.get("prefix_fingerprint") == fingerprint
                            and int(prior.get("prompt_token_count") or 0) == len(token_ids)):
                        try:
                            (store_root / basename).unlink()
                        except OSError:
                            pass
                        return (f"already held as {existing.digest[:12]} "
                                f"({len(token_ids):,} tokens); capture discarded")
                base = f"{args.swap.rstrip('/')}/upstream/{model}"
                obj = AdmittedPath(LlamaCppHTTPAdapter(base, str(store_root)),
                                   sidecar.store()).admit(
                    store_root / basename, model=model, token_ids=token_ids,
                    save_response=saved, prefix_fingerprint=fingerprint)
                # admit() copied the bytes into a digest-named artifact, so the raw
                # capture is now a duplicate of it. Left behind, these accumulate at full
                # size -- 12 of them reached 9.5 GB, more than the artifacts they duplicate
                # -- and nothing ever reads them: lookups go by fingerprint and model, and
                # `auto-<model>-slot<n>-<tokens>.state` carries neither.
                try:
                    (store_root / basename).unlink()
                except OSError as exc:          # a leftover is waste, not a failed admit
                    print(f"[capture] could not remove raw capture {basename}: "
                          f"{str(exc)[:80]}", flush=True)
                return (f"{obj.digest[:12]} covering {len(token_ids):,} tokens "
                        f"(prefix {fingerprint[:12]})")

            loop = CaptureLoop(args.swap, args.store_root,
                               min_tokens=args.capture_min_tokens,
                               interval=args.capture_interval,
                               restorer=restore_on_load,
                               admitter=admit_capture,
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
