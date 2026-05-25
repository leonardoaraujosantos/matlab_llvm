import jail


def test_jail_none_is_passthrough(monkeypatch):
    monkeypatch.setattr(jail.settings, "sandbox_backend", "none")
    argv = ["matlabc", "-repl"]
    assert jail.wrap(argv, "/ws") == argv


def test_jail_bwrap_builds_argv(monkeypatch):
    monkeypatch.setattr(jail.settings, "sandbox_backend", "bwrap")
    monkeypatch.setattr(jail.settings, "sandbox_allow_net", False)
    monkeypatch.setattr(jail.shutil, "which", lambda _t: "/usr/bin/bwrap")
    out = jail.wrap(["matlabc", "-repl"], "/ws")
    assert out[0] == "bwrap"
    assert "--bind" in out and "/ws" in out
    assert "--unshare-net" in out
    assert out[-2:] == ["matlabc", "-repl"]


def test_jail_bwrap_isolation_flags_complete(monkeypatch):
    """The flags that *are* the security boundary — regression-pin them."""
    monkeypatch.setattr(jail.settings, "sandbox_backend", "bwrap")
    monkeypatch.setattr(jail.settings, "sandbox_allow_net", False)
    monkeypatch.setattr(jail.shutil, "which", lambda _t: "/usr/bin/bwrap")
    out = jail.wrap(["matlabc", "-repl"], "/ws")
    pairs = list(zip(out, out[1:]))
    # Root is mounted read-only — host fs visible but un-writable.
    assert ("--ro-bind", "/") in pairs
    # A fresh procfs view (only the sandbox's PIDs are visible).
    assert ("--proc", "/proc") in pairs
    # Minimal /dev — no /dev/mem etc.
    assert ("--dev", "/dev") in pairs
    # /tmp is a clean tmpfs.
    assert ("--tmpfs", "/tmp") in pairs
    # Workspace is the *only* read-write bind.
    assert ("--bind", "/ws") in pairs
    assert ("--chdir", "/ws") in pairs
    # Namespaces — process, IPC, UTS, network. PID namespace alone reduces
    # /proc to the sandbox's own processes.
    for flag in ("--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net"):
        assert flag in out, f"missing isolation flag: {flag}"
    # Lifecycle hardening.
    assert "--die-with-parent" in out
    assert "--new-session" in out
    # argv is appended after the `--` separator.
    sep = out.index("--")
    assert out[sep + 1 :] == ["matlabc", "-repl"]


def test_jail_allow_net_drops_net_unshare(monkeypatch):
    monkeypatch.setattr(jail.settings, "sandbox_backend", "bwrap")
    monkeypatch.setattr(jail.settings, "sandbox_allow_net", True)
    monkeypatch.setattr(jail.shutil, "which", lambda _t: "/usr/bin/bwrap")
    assert "--unshare-net" not in jail.wrap(["m"], "/ws")


def test_jail_missing_tool_falls_back(monkeypatch):
    monkeypatch.setattr(jail.settings, "sandbox_backend", "firejail")
    monkeypatch.setattr(jail.shutil, "which", lambda _t: None)
    argv = ["matlabc"]
    assert jail.wrap(argv, "/ws") == argv


def test_jail_firejail_and_nsjail(monkeypatch):
    monkeypatch.setattr(jail.shutil, "which", lambda t: "/usr/bin/" + t)
    monkeypatch.setattr(jail.settings, "sandbox_allow_net", False)
    for backend in ("firejail", "nsjail"):
        monkeypatch.setattr(jail.settings, "sandbox_backend", backend)
        out = jail.wrap(["m", "-repl"], "/ws")
        assert out[0] == backend
        assert out[-1] == "-repl"


def test_jail_unknown_backend_falls_back(monkeypatch):
    monkeypatch.setattr(jail.settings, "sandbox_backend", "bogus")
    argv = ["m"]
    assert jail.wrap(argv, "/ws") == argv
