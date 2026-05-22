import asyncio
import io

import limits


class _Req:
    def __init__(self, host):
        self.client = type("C", (), {"host": host})()


def test_rate_limit_blocks_after_threshold(monkeypatch):
    monkeypatch.setattr(limits.settings, "rate_limit_per_minute", 3)
    limits._hits.clear()
    req = _Req("1.2.3.4")

    async def go():
        for _ in range(3):
            await limits.rate_limit(req)
        try:
            await limits.rate_limit(req)
            return None
        except Exception as exc:
            return getattr(exc, "status_code", None)

    assert asyncio.run(go()) == 429


def test_rate_limit_disabled(monkeypatch):
    monkeypatch.setattr(limits.settings, "rate_limit_per_minute", 0)
    asyncio.run(limits.rate_limit(_Req("9.9.9.9")))  # must not raise


def test_dir_size(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"de")
    assert limits.dir_size(tmp_path) == 5


def test_would_exceed_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(limits.settings, "user_quota_mb", 0)
    assert limits.would_exceed_quota(tmp_path, 10**9) is False  # disabled
    monkeypatch.setattr(limits.settings, "user_quota_mb", 1)
    (tmp_path / "f").write_bytes(b"x" * 100)
    assert limits.would_exceed_quota(tmp_path, 0) is False
    assert limits.would_exceed_quota(tmp_path, 2 * 1024 * 1024) is True


def test_rate_limit_route_returns_429(client, monkeypatch):
    monkeypatch.setattr(limits.settings, "rate_limit_per_minute", 2)
    limits._hits.clear()
    assert client.post("/v1/check", json={"source": "x=1;"}).status_code == 200
    assert client.post("/v1/check", json={"source": "x=1;"}).status_code == 200
    assert client.post("/v1/check", json={"source": "x=1;"}).status_code == 429


def test_upload_quota_exceeded(client, monkeypatch):
    monkeypatch.setattr(limits, "dir_size", lambda ws: 10**9)  # pretend full
    r = client.post(
        "/v1/files",
        params={"session_id": "quota1"},
        files={"file": ("big.csv", io.BytesIO(b"payload"), "text/csv")},
    )
    assert r.status_code == 413
