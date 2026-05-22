"""Chat endpoint tests (offline / retrieval-only mode).

conftest forces OPENAI_API_KEY="" and indexes a tiny docs corpus, so these
exercise grounding + citations without calling OpenAI.
"""


def test_chat_offline_grounded_with_citations(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [
            {"role": "user", "content": "How do I use fmincon for constrained optimization?"}
        ]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    content = body["choices"][0]["message"]["content"]
    assert "fmincon" in content.lower()
    assert any("optim" in c for c in body["x_citations"])


def test_chat_offline_streaming_sse(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "x", "stream": True, "messages": [
            {"role": "user", "content": "emit systemverilog with -emit-sv"}
        ]},
    )
    assert r.status_code == 200
    assert "data:" in r.text
    assert "[DONE]" in r.text


def test_chat_unmatched_question_still_answers(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "zzz qqq totally unrelated"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["role"] == "assistant"


# --- OpenAI proxy path (mocked httpx so no network/key needed) ------------
class _Resp:
    status_code = 200
    text = ""

    def json(self):
        return {
            "id": "x",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
        }


class _MockClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        return _Resp()

    def stream(self, *a, **k):
        return _MockStream()


class _MockStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_bytes(self):
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"


def test_chat_proxy_non_stream_mocked(client, monkeypatch):
    import routers.chat as chatmod

    monkeypatch.setattr(chatmod.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(chatmod.httpx, "AsyncClient", _MockClient)
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "fmincon?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert "x_citations" in body


def test_chat_proxy_stream_mocked(client, monkeypatch):
    import routers.chat as chatmod

    monkeypatch.setattr(chatmod.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(chatmod.httpx, "AsyncClient", _MockClient)
    r = client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert "[DONE]" in r.text
