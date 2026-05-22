def test_plot_png_default(client):
    r = client.post("/v1/plot", json={"source": "plot([1 2 3])", "session_id": "plt1"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("image/png")
    assert r.content[:4] == b"\x89PNG"


def test_plot_svg_via_query(client):
    r = client.post("/v1/plot?format=svg", json={"source": "plot([1 2])", "session_id": "plt2"})
    assert r.status_code == 200
    assert "svg" in r.headers["content-type"]
    assert b"<svg" in r.content


def test_plot_no_figure_returns_422(client):
    r = client.post("/v1/plot", json={"source": "NOFIG", "session_id": "plt3"})
    assert r.status_code == 422


def test_plot_bad_format_returns_400(client):
    r = client.post("/v1/plot?format=gif", json={"source": "plot(1)", "session_id": "plt4"})
    assert r.status_code == 400
