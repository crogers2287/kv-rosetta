"""An explicit OPAQUE export must be refused unless the runtime can honour it.

Capability discovery is advisory: a caller can skip it and ask for OPAQUE directly. On an
unpatched hybrid runtime that call currently reaches the slot-save POST and returns a plain
sequence-state artifact, which restores successfully and reuses nothing. These tests pin the
refusals at the export boundary itself, offline against a stubbed runtime.

The assertion that no save POST occurred is the load-bearing one. Refusing after the server
has already written a state file still leaks work and still lets a caller believe the
runtime was consulted.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from kv_rosetta.adapters.base import AdapterError, ExportRequest, Representation
from kv_rosetta.adapters.ggsq_envelope import GGSQ_MAGIC, SCKP_MAGIC
from kv_rosetta.adapters.llamacpp_http import LlamaCppHTTPAdapter

PROMPT_TOKENS = (11, 22, 33, 44)


def ggsq_body(tokens=PROMPT_TOKENS, version=3, trailer=b"") -> bytes:
    """A minimally well-formed llama_state_seq_save_file buffer."""
    head = GGSQ_MAGIC + struct.pack("<II", version, len(tokens))
    head += struct.pack(f"<{len(tokens)}i", *tokens)
    return head + b"opaque llama state" * 8 + trailer


def sckp_appendix(n_tokens=252, pos_min=0, pos_max=251, payload=64) -> bytes:
    return (SCKP_MAGIC + struct.pack("<II", 1, 1)
            + struct.pack("<qii", n_tokens, pos_min, pos_max)
            + struct.pack("<Q", payload) + bytes(payload)
            + struct.pack("<Q", 0) + struct.pack("<Q", 0))


def props(**overrides) -> dict:
    base = {
        "slot_checkpoint_persistence": True,
        "slot_checkpoint_format": "sckp/1",
        "sequence_state_version": 3,
        "supports_target_checkpoint_state": True,
        "supports_draft_checkpoint_state": False,
        "supports_speculative_checkpoint_state": False,
        "sckp_serializes_target_state": True,
        "sckp_serializes_draft_state": True,
        "sckp_serializes_speculative_state": True,
        "default_generation_settings": {"type_k": "f16", "n_ctx": 8192},
        "build_info": "bstub-0000000",
        "target_cache_type_k": "f16",
        "target_cache_type_v": "f16",
    }
    base.update(overrides)
    return base


UNPATCHED = {"default_generation_settings": {"type_k": "f16"}, "build_info": "bstub-0000000"}


class StubAdapter(LlamaCppHTTPAdapter):
    """A hybrid runtime that answers exactly what a case needs and records every POST."""

    def __init__(self, runtime_props, slot_dir, *, save=None, body=None, arch="qwen35"):
        super().__init__("http://127.0.0.1:1", str(slot_dir))
        self._props_value = runtime_props
        self._arch = arch
        self._body = ggsq_body() if body is None else body
        self._save = save
        self.posts: list[tuple[str, dict]] = []

    def props(self, refresh: bool = False) -> dict:
        return self._props_value

    def architecture(self, model: str = "") -> str:
        return self._arch

    def _post(self, path: str, payload: dict, timeout=None):
        self.posts.append((path, dict(payload)))
        if "action=save" in path:
            (self.slot_save_path / payload["filename"]).write_bytes(self._body)
            return dict(self._save or {})
        return {}

    def identity(self, model: str = "") -> dict:
        # The cache ABI digest must be the real one, or every cross-runtime identity test
        # compares a constant against itself.
        return {"model_digest": "d" * 64,
                "cache_abi_digest": self.cache_abi_identity(model).digest(),
                "build_info": "bstub-0000000"}

    def model_identity(self, model: str = ""):
        from kv_rosetta.identity import ModelIdentity
        return ModelIdentity(architecture=self._arch, weights_sha256="w" * 64,
                             tokenizer_sha256="t" * 64, chat_template_sha256="c" * 64,
                             adapters=(), notes=())

    def cache_abi_identity(self, model: str = ""):
        # The real implementation, so identity actually varies with what the runtime
        # advertises. A constant here would make every identity test vacuous.
        return LlamaCppHTTPAdapter.cache_abi_identity(self, model)

    def prefix_reuse_support(self, model: str = ""):
        from kv_rosetta import gguf
        return gguf.supports_prefix_reuse(self._arch)


SAVE_OK = {"n_saved": 256, "n_written": 0, "n_checkpoints_saved": 1,
           "checkpoint_bytes": 0, "checkpoint_n_tokens": 252,
           "checkpoint_pos_min": 0, "checkpoint_pos_max": 251}


def save_with_appendix(**overrides) -> tuple[dict, bytes]:
    """A save response whose byte counts describe the body it is paired with."""
    appendix = sckp_appendix()
    body = ggsq_body(trailer=appendix)
    response = dict(SAVE_OK, n_written=len(body), checkpoint_bytes=len(appendix))
    response.update(overrides)
    return response, body


class HybridExportGateTest(unittest.TestCase):
    def build(self, runtime_props, **kwargs):
        directory = Path(tempfile.mkdtemp())
        slots = directory / "slots"
        slots.mkdir()
        adapter = StubAdapter(runtime_props, slots, **kwargs)
        return adapter, directory / "artifact.kvx"

    def refuses(self, runtime_props, **kwargs):
        """Assert an explicit OPAQUE export is refused before any slot-save POST."""
        adapter, out = self.build(runtime_props, **kwargs)
        with self.assertRaises(AdapterError) as caught:
            adapter.export(ExportRequest(model="", out_path=out,
                                         representation=Representation.OPAQUE))
        saves = [p for p, _ in adapter.posts if "action=save" in p]
        self.assertEqual(saves, [], "refused only after asking the server to save")
        self.assertFalse(out.exists(), "refused but still published an artifact")
        return str(caught.exception)

    # -- 1. the unpatched hybrid runtime -------------------------------------------

    def test_unpatched_hybrid_refuses_explicit_opaque_export(self):
        reason = self.refuses(UNPATCHED, save=dict(SAVE_OK))
        self.assertIn("checkpoint", reason.lower())

    def test_non_hybrid_unpatched_runtime_still_exports(self):
        # The gate must key on reusability, not on the absence of a protocol: a plain
        # attention model has always been exportable and must stay that way.
        adapter, out = self.build(UNPATCHED, save=dict(SAVE_OK), arch="qwen2")
        adapter.export(ExportRequest(model="", out_path=out,
                                     representation=Representation.OPAQUE))
        self.assertTrue(out.exists())

    # -- 2. patched-but-not-usable protocols ---------------------------------------

    def test_persistence_false_refuses_export(self):
        self.refuses(props(slot_checkpoint_persistence=False), save=dict(SAVE_OK))

    def test_unknown_format_refuses_export(self):
        self.refuses(props(slot_checkpoint_format="sckp/9"), save=dict(SAVE_OK))

    def test_target_state_false_refuses_export(self):
        self.refuses(props(supports_target_checkpoint_state=False), save=dict(SAVE_OK))

    def test_incomplete_checkpoint_metadata_refuses_export(self):
        for field in ("n_checkpoints_saved", "checkpoint_n_tokens"):
            with self.subTest(field=field):
                self.refuses(props(), save=dict(SAVE_OK, **{field: 0}),
                             body=ggsq_body(trailer=sckp_appendix()))

    # -- 6. the compound tuple allowlist -------------------------------------------

    def test_sequence_version_2_with_sckp_is_refused(self):
        # sckp/1 is proven on sequence version 3 only. Plain ggsq/2 support says nothing
        # about whether an appendix on a version-2 state restores.
        self.refuses(props(sequence_state_version=2), save=dict(SAVE_OK),
                     body=ggsq_body(version=2, trailer=sckp_appendix()))

    # -- 7. active draft/speculative configuration ---------------------------------

    def test_draft_state_required_but_unproven_refuses_export(self):
        self.refuses(props(active_checkpoint_state_classes=["target", "draft"]),
                     save=dict(SAVE_OK), body=ggsq_body(trailer=sckp_appendix()))

    def test_speculative_state_required_but_unproven_refuses_export(self):
        self.refuses(props(active_checkpoint_state_classes=["target", "speculative"]),
                     save=dict(SAVE_OK), body=ggsq_body(trailer=sckp_appendix()))

    def test_target_only_active_configuration_is_allowed(self):
        save, body = save_with_appendix()
        adapter, out = self.build(props(active_checkpoint_state_classes=["target"]),
                                  save=save, body=body)
        adapter.export(ExportRequest(model="", out_path=out,
                                     representation=Representation.OPAQUE))
        self.assertTrue(out.exists())

    # -- 3. incidental SCKP bytes --------------------------------------------------

    def test_incidental_sckp_bytes_are_never_labelled_compound(self):
        # Four bytes of opaque state can spell SCKP. Labelling that artifact compound
        # would tell an importer a hybrid restore is available when none is.
        body = ggsq_body(trailer=b"") 
        body = body[:24] + SCKP_MAGIC + body[24:]
        adapter, out = self.build(props(), save=dict(SAVE_OK, checkpoint_n_tokens=0,
                                                     n_checkpoints_saved=0), body=body)
        with self.assertRaises(AdapterError):
            adapter.export(ExportRequest(model="", out_path=out,
                                         representation=Representation.OPAQUE))

    # -- the declared bounds must agree with the file ------------------------------

    def test_written_byte_count_disagreeing_with_the_file_is_refused(self):
        save, body = save_with_appendix()
        adapter, out = self.build(props(active_checkpoint_state_classes=["target"]),
                                  save=dict(save, n_written=save["n_written"] + 64),
                                  body=body)
        with self.assertRaises(AdapterError) as caught:
            adapter.export(ExportRequest(model="", out_path=out,
                                         representation=Representation.OPAQUE))
        self.assertIn("bytes but", str(caught.exception))
        self.assertFalse(out.exists())

    def test_appendix_not_at_the_declared_offset_is_refused(self):
        # checkpoint_bytes understated by 8 puts the derived offset inside the sequence
        # body, where the magic is not. A scan would still have found the real appendix
        # further on and labelled this compound.
        save, body = save_with_appendix()
        adapter, out = self.build(props(active_checkpoint_state_classes=["target"]),
                                  save=dict(save, checkpoint_bytes=save["checkpoint_bytes"] - 8),
                                  body=body)
        with self.assertRaises(AdapterError) as caught:
            adapter.export(ExportRequest(model="", out_path=out,
                                         representation=Representation.OPAQUE))
        self.assertIn("declared offset", str(caught.exception))

    def test_appendix_trailed_by_extra_bytes_is_refused(self):
        save, body = save_with_appendix()
        adapter, out = self.build(props(active_checkpoint_state_classes=["target"]),
                                  save=dict(save, n_written=len(body) + 8),
                                  body=body + b"leftover")
        with self.assertRaises(AdapterError):
            adapter.export(ExportRequest(model="", out_path=out,
                                         representation=Representation.OPAQUE))

    # -- 8. position zero ----------------------------------------------------------

    def test_checkpoint_pos_min_zero_survives_export(self):
        import json

        from kv_rosetta import container
        save, body = save_with_appendix(checkpoint_pos_min=0)
        adapter, out = self.build(props(active_checkpoint_state_classes=["target"]),
                                  save=save, body=body)
        adapter.export(ExportRequest(model="", out_path=out,
                                     representation=Representation.OPAQUE))
        header = container.read_header(out)
        self.assertEqual(header["coverage"]["checkpoint_pos_min"], 0,
                         "position 0 was rewritten as -1 by an `or` default")
        self.assertIn("+sckp/1", header["coverage"]["format"])
        json.dumps(header)


if __name__ == "__main__":
    unittest.main()


class CapabilityProbeCostTest(unittest.TestCase):
    """Capability discovery writes a slot file, and that must not be mistaken for export.

    state_version() probes the emitted sequence version by saving a slot, because no
    endpoint reports it. Any evidence that an export "refused before any save POST" has to
    measure the export window, not the capability probe that ran before it.
    """

    def build(self, runtime_props):
        directory = Path(tempfile.mkdtemp())
        slots = directory / "slots"
        slots.mkdir()
        save, body = save_with_appendix()
        return StubAdapter(runtime_props, slots, save=save, body=body), directory

    def test_capability_discovery_issues_a_save(self):
        adapter, _ = self.build(props(active_checkpoint_state_classes=["target"]))
        adapter.capabilities()
        self.assertTrue([p for p, _ in adapter.posts if "action=save" in p],
                        "state_version() is documented to probe by saving a slot")

    def test_export_itself_posts_nothing_when_it_refuses(self):
        adapter, directory = self.build(UNPATCHED)
        adapter.posts.clear()
        with self.assertRaises(AdapterError):
            adapter.export(ExportRequest(model="", out_path=directory / "x.kvx",
                                         representation=Representation.OPAQUE))
        self.assertEqual([p for p, _ in adapter.posts], [],
                         "export refused only after contacting the runtime")

    def test_the_probe_file_is_removed(self):
        adapter, _ = self.build(props(active_checkpoint_state_classes=["target"]))
        adapter.capabilities()
        leftover = sorted(p.name for p in adapter.slot_save_path.glob("*probe*"))
        self.assertEqual(leftover, [], f"capability probe left {leftover} behind")
