import auth


def test_auth_enforced_when_token_set(client, monkeypatch):
    monkeypatch.setattr(auth.settings, "api_token", "secret")
    body = {"source": "x = 1;"}
    assert client.post("/v1/check", json=body).status_code == 401
    assert client.post("/v1/check", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert (
        client.post("/v1/check", json=body, headers={"Authorization": "Bearer secret"}).status_code == 200
    )


def test_auth_disabled_by_default(client):
    assert client.post("/v1/check", json={"source": "x = 1;"}).status_code == 200
