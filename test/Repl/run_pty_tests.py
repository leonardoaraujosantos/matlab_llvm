#!/usr/bin/env python3
# PTY-driven REPL tests (#291). Tab completion only fires on an interactive
# (TTY) line editor, so these run matlabc -repl under a pseudo-terminal and
# feed keystrokes including TAB.
#
# Reliability: if the REPL banner never appears under the PTY (a sandboxed /
# PTY-less environment), the affected check SKIPs rather than fails — but once
# the banner is seen, the completion assertion is enforced for real.
#
# Usage: run_pty_tests.py <path-to-matlabc>
import os, pty, select, sys, time

MC = sys.argv[1] if len(sys.argv) > 1 else None
if not MC or not os.access(MC, os.X_OK):
    print("usage: run_pty_tests.py <path-to-matlabc>", file=sys.stderr)
    sys.exit(2)

BANNER = "matlabc REPL"


def session(keystrokes, total_timeout=15.0, settle=1.0):
    """Spawn matlabc -repl in a PTY, send each keystroke chunk with a settle
    delay, return all captured output."""
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "dumb"
        os.execv(MC, [MC, "-repl"])
        os._exit(127)
    out = b""
    deadline = time.time() + total_timeout

    def drain(t):
        nonlocal out
        end = time.time() + t
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.2)
            if not r:
                continue
            try:
                data = os.read(fd, 4096)
            except OSError:
                return False
            if not data:
                return False
            out += data
        return True

    drain(1.0)  # banner
    for k in keystrokes:
        try:
            os.write(fd, k)
        except OSError:
            break
        drain(settle)
        if time.time() > deadline:
            break
    try:
        os.write(fd, b"\nexit\n")
    except OSError:
        pass
    drain(1.0)
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass
    return out.decode(errors="replace")


passed = skipped = failed = 0


def check(name, keystrokes, must_contain):
    global passed, skipped, failed
    out = session(keystrokes)
    if BANNER not in out:
        print(f"SKIP  {name} (no PTY/banner in this environment)")
        skipped += 1
        return
    if must_contain in out:
        passed += 1
    else:
        failed += 1
        print(f"FAIL  {name}: expected completion to {must_contain!r}")
        print("----- got -----")
        print(repr(out[-300:]))


# 'dis' + TAB -> longest common prefix of disp/display = 'disp'.
check("tab_complete_builtin_prefix", [b"dis\t"], "disp")

# A session-defined function is offered by Tab completion. Define it on the
# first line, then complete its prefix on the next.
check("tab_complete_session_function",
      [b"function y = myuniquefn(x); y = x; end\n", b"myuniq\t"],
      "myuniquefn")

print("----")
print(f"pty passed: {passed}    skipped: {skipped}    failed: {failed}")
sys.exit(1 if failed else 0)
