#!/usr/bin/env bash
# Build a llama.cpp with persistent slot checkpoints, reproducibly.
#
# Refuses to run against an unexpected upstream revision or an altered patch. Applying a
# patch "fuzzily" to unknown source is how an experiment stops being reproducible: the
# binary would no longer correspond to anything committed here.
#
# Usage:  scripts/build_patched_llama.sh [target-dir]
set -euo pipefail

# --- pinned inputs ----------------------------------------------------------------------

UPSTREAM_URL="https://github.com/ggml-org/llama.cpp.git"

# The revision this patch has been verified to apply to, by three-way merge.
EXPECTED_BASE="ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc"

# Upstream PR #26004, "server : preserve context checkpoints across slot save/restore".
# Its own stated base is adb55e5148dc93bcdca7212a2d1df3ccc422959a, which differs from
# EXPECTED_BASE - hence the three-way apply below rather than a direct one.
PR_NUMBER="26004"
PATCH_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches/llama.cpp/0001-persist-slot-prompt-checkpoints.patch"
EXPECTED_PATCH_SHA256="baf44e7c06f1a8b16bcc7de1019c2a36e8147b0b32a61b33bff4160e445fc22f"

TARGET="${1:-$HOME/llama.cpp-kvx-patched}"

# Build flags, pinned so a rebuild is comparable to the run that produced the evidence.
CMAKE_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_CUDA=ON
  -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TESTS=OFF
  # ccache is disabled deliberately. On this host ~/.cache/ccache/tmp is owned by root
  # from an earlier privileged build, so an unprivileged compile dies with "failed to
  # create temporary file ... Permission denied". A reproducible build must not depend on
  # the ownership of a shared cache directory it does not control.
  -DGGML_CCACHE=OFF
)

say() { printf '  %s\n' "$*"; }
die() { printf 'REFUSING: %s\n' "$*" >&2; exit 1; }

# --- verify the patch is the one we committed --------------------------------------------

[ -f "$PATCH_FILE" ] || die "patch not found: $PATCH_FILE"
actual_sha="$(sha256sum "$PATCH_FILE" | cut -d' ' -f1)"
[ "$actual_sha" = "$EXPECTED_PATCH_SHA256" ] || die \
  "patch sha256 mismatch
     expected $EXPECTED_PATCH_SHA256
     actual   $actual_sha
   The committed patch has been modified. Update EXPECTED_PATCH_SHA256 deliberately, or
   restore the file - do not build from an unverified patch."
say "patch sha256 ok: ${actual_sha:0:16}..."

# --- obtain the exact upstream revision --------------------------------------------------

if [ -d "$TARGET/.git" ]; then
  say "reusing $TARGET"
  git -C "$TARGET" fetch --quiet origin "$EXPECTED_BASE" || true
else
  say "cloning upstream into $TARGET (full history: a shallow clone cannot resolve the base)"
  git clone --quiet "$UPSTREAM_URL" "$TARGET"
fi

# A shallow clone silently fails ancestry checks rather than erroring, which is how an
# unverified base slips through. ~/llama.cpp on this host is shallow (151 commits) for
# exactly that reason.
if [ -f "$TARGET/.git/shallow" ]; then
  say "target is shallow; unshallowing so the base commit can be verified"
  git -C "$TARGET" fetch --quiet --unshallow || git -C "$TARGET" fetch --quiet --depth=1000
fi

git -C "$TARGET" cat-file -e "${EXPECTED_BASE}^{commit}" 2>/dev/null || die \
  "expected base $EXPECTED_BASE is not present in $TARGET after fetching"

git -C "$TARGET" checkout --quiet --detach "$EXPECTED_BASE"
head_sha="$(git -C "$TARGET" rev-parse HEAD)"
[ "$head_sha" = "$EXPECTED_BASE" ] || die "checked out $head_sha, expected $EXPECTED_BASE"
say "upstream pinned at ${EXPECTED_BASE:0:9}"

# --- apply -------------------------------------------------------------------------------

git -C "$TARGET" fetch --quiet origin "pull/${PR_NUMBER}/head:pr${PR_NUMBER}" 2>/dev/null || true

if git -C "$TARGET" apply --check "$PATCH_FILE" 2>/dev/null; then
  git -C "$TARGET" apply "$PATCH_FILE"
  say "patch applied directly"
elif git -C "$TARGET" apply -3 --check "$PATCH_FILE" 2>/dev/null; then
  git -C "$TARGET" apply -3 "$PATCH_FILE"
  say "patch applied by three-way merge (PR base differs from EXPECTED_BASE)"
else
  die "patch does not apply to $EXPECTED_BASE, directly or by three-way merge.
   Rebase it deliberately and update EXPECTED_PATCH_SHA256 - never apply with fuzz."
fi

# --- build -------------------------------------------------------------------------------

say "configuring: ${CMAKE_FLAGS[*]}"
cmake -S "$TARGET" -B "$TARGET/build" "${CMAKE_FLAGS[@]}" >/dev/null
say "building llama-server (this takes a while)"
cmake --build "$TARGET/build" --target llama-server -j "$(nproc)" >/dev/null

BIN="$TARGET/build/bin/llama-server"
[ -x "$BIN" ] || die "build produced no llama-server at $BIN"

# --- probe -------------------------------------------------------------------------------

say "built: $BIN"
say "upstream base:  $EXPECTED_BASE"
say "patch sha256:   $EXPECTED_PATCH_SHA256"
say "state seq ver:  $(grep -oE '#define LLAMA_STATE_SEQ_VERSION +[0-9]+' "$TARGET/include/llama.h" | grep -oE '[0-9]+$')"
# The server implementation lives in libllama-server-impl.so; llama-server itself is a thin
# launcher (~18 KB), so probing the executable for the magic finds nothing and proves nothing.
IMPL="$TARGET/build/bin/libllama-server-impl.so"
if [ -f "$IMPL" ]; then
  n_magic="$(strings "$IMPL" 2>/dev/null | grep -c 'SCKP' || true)"
  n_ckpt="$(strings "$IMPL" 2>/dev/null | grep -c 'context checkpoint' || true)"
  say "patch compiled in: SCKP magic x$n_magic, 'context checkpoint' strings x$n_ckpt (in $(basename "$IMPL"))"
  [ "$n_magic" -gt 0 ] || die "built binary does not contain the checkpoint magic; the patch did not compile in"
else
  die "expected $IMPL; cannot verify the patch compiled in"
fi

cat <<EOF

  Startup flags used by the live tests:

    $BIN --model <MODEL.gguf> --host 127.0.0.1 --port 8785 \\
      -ngl 99 -c 8192 --parallel 1 -fa on \\
      --slot-save-path <SLOT_DIR>/ --no-warmup

  Before trusting any positive result, run the negative control against this binary and
  confirm it now FAILS - an unpatched build must still show cache_n=0:

    KVX_HYBRID_URL=http://127.0.0.1:8785 KVX_HYBRID_SLOTS=<SLOT_DIR> \\
      python3 -m unittest tests.test_hybrid_negative_control -v
EOF
