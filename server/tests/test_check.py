def test_check_ok(client):
    r = client.post("/v1/check", json={"source": "x = 1 + 1;"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_check_error_yields_diagnostic(client):
    r = client.post("/v1/check", json={"source": "ERR not valid"})
    body = r.json()
    assert body["ok"] is False
    assert any(d["severity"] == "error" for d in body["diagnostics"])
