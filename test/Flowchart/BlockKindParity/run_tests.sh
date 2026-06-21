#!/usr/bin/env bash
# Editor <-> simulator block-kind parity guard (#343 §1).
#
# The simulator/lowering registers a set of `signal_*` block kinds (queried via
# `matlabc -simulate --list-supported-kinds`). The IDE editor must expose the
# same kinds in its block library, or a model placeable in the editor won't
# simulate (and vice-versa). The two live in separate repos, so this guard owns
# the in-repo half: it diffs the live registered-kind set against a committed
# snapshot. Adding or removing a kind fails this test until the snapshot is
# updated — the reviewer's cue to mirror the change in the editor's NodeKind.
#
# To update after intentionally adding/removing a kind:
#   matlabc -simulate --list-supported-kinds | \
#     grep -oE '"kind": "[a-z_0-9]+"' | sed -E 's/.*"([a-z_0-9]+)"/\1/' | \
#     sort -u > test/Flowchart/BlockKindParity/registered_block_kinds.txt
#   ... and add the matching NodeKind in the IDE repo.
#
# Usage: run_tests.sh <path-to-matlabc>
set -u
export LC_ALL=C

MATLABC="${1:-}"
if [[ -z "$MATLABC" || ! -x "$MATLABC" ]]; then
  echo "usage: $0 <path-to-matlabc>" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
SNAPSHOT="$HERE/registered_block_kinds.txt"
[[ -f "$SNAPSHOT" ]] || { echo "FAIL: missing snapshot $SNAPSHOT" >&2; exit 1; }

LIVE="$("$MATLABC" -simulate --list-supported-kinds 2>/dev/null \
        | grep -oE '"kind": "[a-z_0-9]+"' \
        | sed -E 's/.*"([a-z_0-9]+)"/\1/' | sort -u)"

if [[ -z "$LIVE" ]]; then
  echo "FAIL: --list-supported-kinds produced no kinds" >&2
  exit 1
fi

if diff -u "$SNAPSHOT" <(printf '%s\n' "$LIVE") >/dev/null; then
  echo "PASS: $(printf '%s\n' "$LIVE" | wc -l | tr -d ' ') registered block kinds match the snapshot"
  exit 0
fi

echo "FAIL: registered signal_* block kinds drifted from the committed snapshot." >&2
echo "      (+ = newly registered, must be mirrored in the IDE editor NodeKind;" >&2
echo "       - = removed, drop it from the editor too)" >&2
diff -u "$SNAPSHOT" <(printf '%s\n' "$LIVE") | grep -E '^[+-]signal_' | sed 's/^/  /' >&2
echo "      Update $SNAPSHOT once the editor is in sync." >&2
exit 1
