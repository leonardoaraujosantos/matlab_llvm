def test_plot_png_default_json(client):
    # Default response is JSON (consistent with the OpenAPI declaration).
    r = client.post("/v1/plot", json={"source": "plot([1 2 3])", "session_id": "plt1"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["ok"] is True
    assert body["format"] == "png"
    assert len(body["artifacts"]) == 1
    assert body["artifacts"][0].endswith(".png")


def test_plot_artifact_downloadable_via_files(client):
    r = client.post("/v1/plot", json={"source": "plot([1 2 3])", "session_id": "plt_dl"})
    assert r.status_code == 200, r.text
    art = r.json()["artifacts"][0]
    # Same session_id resolves the same workspace the figure was written into.
    f = client.get(f"/v1/files/{art}", params={"session_id": "plt_dl"})
    assert f.status_code == 200, f.text
    assert f.content[:4] == b"\x89PNG"


def test_plot_raw_query_streams_bytes(client):
    r = client.post("/v1/plot?raw=true", json={"source": "plot([1 2 3])", "session_id": "plt_raw"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/png")
    assert r.content[:4] == b"\x89PNG"


def test_plot_accept_header_streams_bytes(client):
    r = client.post(
        "/v1/plot",
        json={"source": "plot([1 2 3])", "session_id": "plt_acc"},
        headers={"accept": "image/png"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/png")
    assert r.content[:4] == b"\x89PNG"


def test_plot_svg_via_query_raw(client):
    r = client.post("/v1/plot?format=svg&raw=true", json={"source": "plot([1 2])", "session_id": "plt2"})
    assert r.status_code == 200
    assert "svg" in r.headers["content-type"]
    assert b"<svg" in r.content


def test_plot_svg_via_query_json(client):
    r = client.post("/v1/plot?format=svg", json={"source": "plot([1 2])", "session_id": "plt2j"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["format"] == "svg"
    assert body["artifacts"][0].endswith(".svg")


def test_plot_no_figure_returns_422(client):
    r = client.post("/v1/plot", json={"source": "NOFIG", "session_id": "plt3"})
    assert r.status_code == 422


def test_plot_bad_format_returns_400(client):
    r = client.post("/v1/plot?format=gif", json={"source": "plot(1)", "session_id": "plt4"})
    assert r.status_code == 400
