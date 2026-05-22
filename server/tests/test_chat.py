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
