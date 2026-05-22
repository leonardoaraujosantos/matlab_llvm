import pytest


@pytest.mark.parametrize(
    "target,needle",
    [
        ("python", "print"),
        ("typescript", "console.log"),
        ("c", "int main"),
        ("cpp", "int main"),
        ("systemverilog", "module"),
    ],
)
def test_codegen_targets(client, target, needle):
    r = client.post(f"/v1/codegen/{target}", json={"source": "y = 2;"})
    body = r.json()
    assert body["ok"] is True
    assert body["language"] == target
    assert needle in body["code"]


def test_codegen_unknown_target_404(client):
    r = client.post("/v1/codegen/rust", json={"source": "y = 2;"})
    assert r.status_code == 404


def test_codegen_error_has_no_code(client):
    r = client.post("/v1/codegen/python", json={"source": "ERR"})
    body = r.json()
    assert body["ok"] is False
    assert body["code"] == ""
