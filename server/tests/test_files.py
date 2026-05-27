import io


def test_upload_list_download_roundtrip(client):
    content = b"1,2,3\n4,5,6\n"
    r = client.post(
        "/v1/files",
        params={"session_id": "files1"},
        files={"file": ("data.csv", io.BytesIO(content), "text/csv")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["file"]["path"] == "data.csv"

    r = client.get("/v1/files", params={"session_id": "files1"})
    assert "data.csv" in [f["path"] for f in r.json()["files"]]

    r = client.get("/v1/files/data.csv", params={"session_id": "files1"})
    assert r.status_code == 200
    assert r.content == content


def test_files_sees_live_stateful_session_workspace(client, tmp_path, monkeypatch):
    """Issue #55: /v1/files must resolve to the live session workspace (the
    worker's adopted pool dir), so artifacts produced by a stateful REPL turn
    are listable/downloadable before the session is retired. Without the fix the
    router looked only at the deterministic dir and 404'd."""
    import services

    live = tmp_path / "live_pool_dir"
    live.mkdir()
    (live / "plot.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Simulate a stateful session that has adopted `live` as its workspace.
    monkeypatch.setattr(
        services.MANAGER,
        "workspace_of",
        lambda user_id, session_id: live if session_id == "live55" else None,
    )

    r = client.get("/v1/files", params={"session_id": "live55"})
    assert r.status_code == 200, r.text
    assert "plot.png" in [f["path"] for f in r.json()["files"]]

    r = client.get("/v1/files/plot.png", params={"session_id": "live55"})
    assert r.status_code == 200, r.text
    assert r.content[:4] == b"\x89PNG"

    # The deterministic dir for this session holds no such file — proving the
    # listing came from the live workspace, not the old det-only path.
    assert not (services.workspace_for(None, "live55") / "plot.png").exists()


def test_path_traversal_rejected(client):
    r = client.get(
        "/v1/files/..%2F..%2Fetc%2Fpasswd", params={"session_id": "files1"}
    )
    assert r.status_code in (400, 404)
