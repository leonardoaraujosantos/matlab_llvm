def test_repl_echo(client):
    r = client.post("/v1/repl", json={"source": "disp(42)"})
    body = r.json()
    assert body["ok"] is True
    assert "ans =" in body["stdout"]


def test_repl_timeout(client):
    r = client.post("/v1/repl", json={"source": "INFLOOP"})
    body = r.json()
    assert body["timed_out"] is True
    assert body["ok"] is False


def test_repl_captures_figure_artifact(client):
    r = client.post("/v1/repl", json={"source": "PLOT a curve", "session_id": "plotsess"})
    body = r.json()
    assert body["ok"] is True
    assert any(a.endswith(".png") for a in body["artifacts"])
