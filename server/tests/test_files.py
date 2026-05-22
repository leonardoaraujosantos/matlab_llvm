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


def test_path_traversal_rejected(client):
    r = client.get(
        "/v1/files/..%2F..%2Fetc%2Fpasswd", params={"session_id": "files1"}
    )
    assert r.status_code in (400, 404)
