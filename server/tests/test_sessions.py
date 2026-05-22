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
