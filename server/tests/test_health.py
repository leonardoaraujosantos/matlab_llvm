def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "matlabc" in body


def test_healthz_reports_sandbox_state(client):
    """/healthz exposes the configured tier-2 sandbox so ops + tests can verify it."""
    r = client.get("/healthz")
    body = r.json()
    assert "sandbox" in body
    sb = body["sandbox"]
    assert sb["backend"] in ("none", "bwrap", "firejail", "nsjail")
    assert isinstance(sb["active"], bool)
    assert isinstance(sb["allow_net"], bool)


def test_healthz_sandbox_active_when_tool_present(client, monkeypatch):
    """When backend is set AND the tool resolves on PATH, ``active`` flips true."""
    import shutil
    from config import settings

    monkeypatch.setattr(settings, "sandbox_backend", "bwrap")
    monkeypatch.setattr(shutil, "which", lambda _t: "/usr/bin/bwrap")
    r = client.get("/healthz")
    sb = r.json()["sandbox"]
    assert sb["backend"] == "bwrap"
    assert sb["active"] is True


def test_healthz_sandbox_inactive_when_tool_missing(client, monkeypatch):
    import shutil
    from config import settings

    monkeypatch.setattr(settings, "sandbox_backend", "bwrap")
    monkeypatch.setattr(shutil, "which", lambda _t: None)
    r = client.get("/healthz")
    sb = r.json()["sandbox"]
    assert sb["backend"] == "bwrap"
    assert sb["active"] is False
