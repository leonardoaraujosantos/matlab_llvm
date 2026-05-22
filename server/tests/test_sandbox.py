import asyncio
import sys

import sandbox


def _run(coro):
    return asyncio.run(coro)


def test_normal_output(tmp_path):
    res = _run(sandbox.run([sys.executable, "-c", "print('hi')"], cwd=tmp_path))
    assert res.ok
    assert "hi" in res.stdout


def test_wall_timeout_kills_child(tmp_path):
    res = _run(
        sandbox.run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            timeout=1,
        )
    )
    assert res.timed_out
    assert not res.ok


def test_output_is_capped(tmp_path):
    res = _run(
        sandbox.run(
            [sys.executable, "-c", "print('x' * 100000)"],
            cwd=tmp_path,
            output_cap=1000,
        )
    )
    assert res.stdout_truncated
    assert len(res.stdout) <= 1000


def test_env_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("SECRET_LEAK", "should-not-pass")
    res = _run(
        sandbox.run(
            [sys.executable, "-c", "import os; print(os.environ.get('SECRET_LEAK', 'ABSENT'))"],
            cwd=tmp_path,
        )
    )
    assert "ABSENT" in res.stdout
