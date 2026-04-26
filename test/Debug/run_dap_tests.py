#!/usr/bin/env python3
"""Driver: run every scn_* in dap_scenarios.py against matlabc -dap."""

import os
import sys
import traceback


def main():
    if len(sys.argv) != 2:
        print("usage: run_dap_tests.py <path-to-matlabc>", file=sys.stderr)
        return 2
    matlabc = sys.argv[1]
    if not os.path.isfile(matlabc) or not os.access(matlabc, os.X_OK):
        print(f"matlabc not executable: {matlabc}", file=sys.stderr)
        return 2

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    program = os.path.join(here, "dap_program.m")

    import dap_scenarios

    passed = 0
    failed = []
    for name, fn in dap_scenarios.all_scenarios():
        sys.stdout.write(f"  {name} ... ")
        sys.stdout.flush()
        try:
            fn(matlabc, program)
        except Exception as e:
            print("FAIL")
            failed.append((name, e, traceback.format_exc()))
            continue
        print("ok")
        passed += 1

    print("----")
    print(f"passed: {passed}    failed: {len(failed)}")
    for name, _, tb in failed:
        print(f"\n=== {name} ===")
        print(tb)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
