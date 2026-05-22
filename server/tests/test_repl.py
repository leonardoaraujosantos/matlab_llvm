def test_repl_echo(client):
    r = client.post("/v1/repl", json={"source": "disp(7)"})
    body = r.json()
    assert body["ok"] is True
    assert "7" in body["stdout"]


def test_repl_timeout(client):
    r = client.post("/v1/repl", json={"source": "INFLOOP", "session_id": "to1"})
    body = r.json()
    assert body["timed_out"] is True
    assert body["ok"] is False


def test_repl_captures_figure_artifact(client):
    r = client.post("/v1/repl", json={"source": "PLOT a curve", "session_id": "plotsess"})
    body = r.json()
    assert body["ok"] is True
    assert any(a.endswith(".png") for a in body["artifacts"])


def test_repl_stateful_persists_variables(client):
    s = "persist1"
    r1 = client.post("/v1/repl", json={"source": "x = 5;", "session_id": s})
    assert r1.json()["ok"] is True
    r2 = client.post("/v1/repl", json={"source": "disp(x)", "session_id": s})
    body = r2.json()
    assert body["stateful"] is True
    assert "5" in body["stdout"]


def test_repl_stateless_does_not_persist(client):
    s = "nostate1"
    client.post("/v1/repl", json={"source": "x = 5;", "session_id": s, "stateful": False})
    r2 = client.post("/v1/repl", json={"source": "disp(x)", "session_id": s, "stateful": False})
    body = r2.json()
    assert body["stateful"] is False
    assert "5" not in body["stdout"]
