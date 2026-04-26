#!/usr/bin/env python3
# Extract the line operand of every matlab_dbg_hook call from an MLIR
# dump (read from stdin), one number per output line, in the order the
# calls appear.
#
# The lowering emits each hook as:
#   %a = arith.constant <file_id> : i32
#   %b = arith.constant <line>    : i32
#   %_ = "matlab.call_builtin"(%a, %b) {callee = "matlab_dbg_hook"} ...
#
# We build a map of SSA name -> integer constant from every
# `arith.constant N : i32` definition we see, then for each hook call
# look up the second operand and print its value.
import re
import sys

CONST_RE = re.compile(
    r'^\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i32\b'
)
HOOK_RE = re.compile(
    r'"matlab\.call_builtin"\(\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*,'
    r'\s*(%[A-Za-z_][A-Za-z0-9_]*)\s*\)[^\n]*callee\s*=\s*"matlab_dbg_hook"'
)

vals = {}
for raw in sys.stdin:
    m = CONST_RE.match(raw)
    if m:
        vals[m.group(1)] = int(m.group(2))
    h = HOOK_RE.search(raw)
    if h:
        line_op = h.group(2)
        if line_op not in vals:
            sys.stderr.write(
                f"extract_hook_lines: hook line operand {line_op} "
                f"has no preceding arith.constant definition\n"
            )
            sys.exit(2)
        print(vals[line_op])
