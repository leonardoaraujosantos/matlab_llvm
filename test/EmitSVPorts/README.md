# EmitSVPorts — fi-spec ↔ SV declaration regression suite

Each `<name>.m` is a small synthesizable function whose ports
declare a known fi spec via either:

- `% hdl: port(name, fi, signed|unsigned, W, F)` pragma
- typed driver call `func(fi(value, sign, W, F))`

The matching `<name>.expected` file lists *substring* matches that
must appear in the emitted SV. Each substring is one line. The
suite passes when every substring appears at least once.

This is intentionally substring-based (not full golden diff) so
the assertions stay focused on the property under test
(signedness + bit width on port and register declarations) and
don't have to be regenerated whenever some unrelated emitter
detail changes.

Asserted shapes:

- `input  logic [W-1:0] <name>` — unsigned input port
- `input  logic signed [W-1:0] <name>` — signed input port
- `output logic [W-1:0] <name>` / `output logic signed [W-1:0] <name>`
- `logic [W-1:0] <reg>;` / `logic signed [W-1:0] <reg>;` — persistent register

Coverage matrix (extend as needed):

|        | 4 | 8 | 16 | 32 |
|--------|---|---|----|----|
| signed   | ☐ | ☑ | ☑ | ☑ |
| unsigned | ☑ | ☑ | ☑ | ☐ |

Run via `run_tests.sh <path-to-matlabc>` or via ctest
(`ctest -R emit-sv-ports`).
