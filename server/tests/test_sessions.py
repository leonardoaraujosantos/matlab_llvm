import asyncio

import sessions
from workspaces import workspace_for


def _run(coro):
    return asyncio.run(coro)


def test_session_persists_then_evicts():
    mgr = sessions.SessionManager()
    ws = workspace_for("u", "sess_unit")

    async def go():
        r1 = await mgr.run_turn("u", "sess_unit", ws, "x = 9;", timeout=3)
        r2 = await mgr.run_turn("u", "sess_unit", ws, "disp(x)", timeout=3)
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
    ws = workspace_for("u", "sess_to")

    async def go():
        res = await mgr.run_turn("u", "sess_to", ws, "INFLOOP", timeout=1)
        active = mgr.active
        await mgr.shutdown()
        return res, active

    res, active = _run(go())
    assert res["timed_out"] is True
    assert active == 0  # a timed-out session is recycled
