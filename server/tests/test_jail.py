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
