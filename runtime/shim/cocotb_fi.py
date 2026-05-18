"""Fixed-point pack / unpack helpers for the matlabc CocoTB harness.

Bridges the Python reference model (real-valued, runs in matlab_runtime.py
fi semantics) and the SV DUT (raw bit vectors on signed/unsigned ports).
Generated `test_<name>.py` testbenches import this module to convert
between the two representations cycle by cycle.

Conventions
-----------
- All packed values are non-negative Python ints with WL bits set; the
  caller knows which signedness applies and uses unpack_fi accordingly.
- Saturating, not wrapping. uint8(-5) packs to 0; int8(200) packs to 127.
  Matches the matlab_runtime saturating int casts the LLVM / C / Python
  emit paths already use.
- FL = number of fractional bits. real_value = bits / 2**FL with sign
  applied per the WL signedness.
"""


def pack_fi(value, signed: bool, wl: int, fl: int) -> int:
    """Real value -> packed WL-bit integer, ready to write into a CocoTB
    signal. `value` may be a Python int or float; it is rounded to the
    nearest integer in the Q(WL-FL).FL grid then saturated."""
    if wl <= 0:
        raise ValueError(f"pack_fi: WL must be positive, got {wl}")
    raw = round(float(value) * (1 << fl))
    if signed:
        lo = -(1 << (wl - 1))
        hi = (1 << (wl - 1)) - 1
    else:
        lo = 0
        hi = (1 << wl) - 1
    if raw < lo:
        raw = lo
    elif raw > hi:
        raw = hi
    if signed and raw < 0:
        raw += 1 << wl  # two's complement encoding for the signal
    return raw


def unpack_fi(bits, signed: bool, wl: int, fl: int) -> float:
    """Packed WL-bit integer -> real value. Accepts either a Python int
    or a CocoTB BinaryValue / LogicArray-shaped object — the latter is
    coerced via int() before sign-extending. Inputs with X / Z bits
    raise ValueError so the testbench fails loudly instead of silently
    masking a bad sample."""
    try:
        raw = int(bits)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unpack_fi: signal value not resolvable: {bits!r}") from exc
    if wl <= 0:
        raise ValueError(f"unpack_fi: WL must be positive, got {wl}")
    raw &= (1 << wl) - 1  # strip any spurious upper bits
    if signed and (raw & (1 << (wl - 1))):
        raw -= 1 << wl
    return raw / (1 << fl)


def fi_range(signed: bool, wl: int, fl: int):
    """(lo, hi) real-value range covered by a port. Used by the random
    vector generator so it stays inside the legal grid (otherwise
    pack_fi saturates and we lose coverage on the upper end)."""
    if signed:
        lo_raw = -(1 << (wl - 1))
        hi_raw = (1 << (wl - 1)) - 1
    else:
        lo_raw = 0
        hi_raw = (1 << wl) - 1
    scale = 1 << fl
    return lo_raw / scale, hi_raw / scale
