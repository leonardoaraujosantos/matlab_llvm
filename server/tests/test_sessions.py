import asyncio

import sessions
from config import settings


def _run(coro):
    return asyncio.run(coro)


def test_session_persists_then_evicts():
    mgr = sessions.SessionManager()

    async def go():
        r1 = await mgr.run_turn("u", "sess_unit", "x = 9;", timeout=3)
        r2 = await mgr.run_turn("u", "sess_unit", "disp(x)", timeout=3)
        active = mgr.active
        evicted = await mgr.evict_idle(max_idle_s=-1.0)  # force-evict all
        await mgr.shutdown()
        return r1, r2, active, evicted

    r1, r2, active, evicted = _run(go())
    assert r1["alive"] is True
    assert "9" in r2["stdout"]
    assert active == 1
    assert evicted == 1


def test_session_timeout_drops_session():
    mgr = sessions.SessionManager()

    async def go():
        res = await mgr.run_turn("u", "sess_to", "INFLOOP", timeout=1)
        active = mgr.active
        await mgr.shutdown()
        return res, active

    res, active = _run(go())
    assert res["timed_out"] is True
    assert active == 0  # a timed-out session is recycled


def test_warm_pool_adopts_dir_and_migrates_files(monkeypatch):
    monkeypatch.setattr(settings, "warm_pool_size", 2)
    mgr = sessions.SessionManager()

    async def go():
        await mgr.pool.fill(2)
        size = mgr.pool.size
        det = sessions.workspace_for("wu", "wsess")
        (det / "staged.txt").write_text("hi")  # stage a file before the session
        await mgr.run_turn("wu", "wsess", "x = 3;", timeout=3)
        ws = mgr.workspace_of("wu", "wsess")
        adopted = ".pool" in str(ws)
        migrated_in = (ws / "staged.txt").exists()
        r2 = await mgr.run_turn("wu", "wsess", "disp(x)", timeout=3)
        await mgr.shutdown()
        migrated_back = (det / "staged.txt").exists()
        return size, adopted, migrated_in, r2["stdout"], migrated_back

    size, adopted, migrated_in, out2, migrated_back = _run(go())
    assert size == 2
    assert adopted is True
    assert migrated_in is True
    assert "3" in out2
    assert migrated_back is True


def test_stateful_artifact_visible_via_files_resolver(monkeypatch):
    """Issue #55: an artifact written during a *live* stateful turn lands in the
    worker's adopted pool dir, not the deterministic dir. The session-aware
    resolver that /v1/files* uses must surface it before the session is retired
    (not only after migrate-back on eviction)."""
    monkeypatch.setattr(settings, "warm_pool_size", 2)
    import services

    async def go():
        await sessions.MANAGER.pool.fill(2)
        det = sessions.workspace_for(None, "issue55")
        # Stateful turn writes a figure into the worker's cwd (the pool dir).
        await sessions.MANAGER.run_turn(None, "issue55", "saveas(gcf, 'plot.png')", timeout=3)
        ws = sessions.MANAGER.workspace_of(None, "issue55")
        # The mismatch condition holds: live ws is an adopted pool dir != det.
        adopted = ".pool" in str(ws) and ws.resolve() != det.resolve()
        # /v1/files resolution path must point at the live ws and see the file.
        resolved_to_ws = services.resolve_workspace(None, "issue55").resolve() == ws.resolve()
        listed = [f["path"] for f in services.list_workspace(session_id="issue55")]
        data = services.read_workspace_file("plot.png", session_id="issue55")
        # And the deterministic dir does NOT yet hold it (proving the old path 404s).
        det_has_it = (det / "plot.png").exists()
        await sessions.MANAGER.shutdown()
        return adopted, resolved_to_ws, listed, data, det_has_it

    adopted, resolved_to_ws, listed, data, det_has_it = _run(go())
    assert adopted is True
    assert resolved_to_ws is True
    assert "plot.png" in listed
    assert data[:4] == b"\x89PNG"
    assert det_has_it is False


def test_warm_pool_refills_after_acquire(monkeypatch):
    monkeypatch.setattr(settings, "warm_pool_size", 2)
    mgr = sessions.SessionManager()

    async def go():
        await mgr.pool.fill(2)
        await mgr.run_turn("r", "s", "y = 1;", timeout=3)  # consumes one worker
        # request_fill scheduled a refill; let it run
        await asyncio.sleep(0.2)
        size = mgr.pool.size
        await mgr.shutdown()
        return size

    # pool should have been topped back up to warm_pool_size (2)
    assert _run(go()) == 2
