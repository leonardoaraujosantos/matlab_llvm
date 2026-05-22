"""POST /v1/chat/completions — OpenAI-compatible chat grounded in the docs.

Retrieves top-k doc chunks for the user's last message and prepends a cited
context block as a system message. With an OpenAI key it proxies upstream
(streaming passthrough); without one it returns a retrieval-only answer so the
endpoint is functional offline.
"""

from __future__ import annotations

import json
import time
import uuid

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

import rag
from config import settings
from models import ChatCompletionRequest

router = APIRouter(prefix="/v1", tags=["chat"])


def _last_user(messages) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return messages[-1].content if messages else ""


def _context_block(hits) -> tuple[str, list[str]]:
    if not hits:
        system = (
            "You are an assistant for the matlab_llvm MATLAB-subset compiler. "
            "No local documentation matched this question; answer from general "
            "knowledge and note any uncertainty."
        )
        return system, []
    parts = [
        "You are a documentation-grounded assistant for the matlab_llvm "
        "MATLAB-subset compiler. Answer using the cited context below and cite "
        "sources inline as [source]. If the context is insufficient, say so.",
        "",
        "=== CONTEXT ===",
    ]
    citations: list[str] = []
    for _score, ch in hits:
        parts.append(f"[{ch.source}] {ch.title}\n{ch.text}\n")
        if ch.source not in citations:
            citations.append(ch.source)
    parts.append("=== END CONTEXT ===")
    return "\n".join(parts), citations


def _completion_obj(model: str, content: str, citations: list[str]) -> dict:
    return {
        "id": f"chatcmpl-local-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_citations": citations,
    }


def _offline_answer(hits, citations: list[str]) -> str:
    if not hits:
        return (
            "No OpenAI key is configured and no local documentation matched your "
            "question. Set OPENAI_API_KEY to enable full answers."
        )
    top = hits[0][1]
    return (
        "No OpenAI key is configured, so here is the most relevant documentation "
        f"(retrieval-only). Sources: {', '.join(citations)}.\n\n"
        f"## {top.title} [{top.source}]\n{top.text}"
    )


def _offline_stream(model: str, content: str):
    cid = f"chatcmpl-local-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def frame(delta: dict, finish=None) -> str:
        return "data: " + json.dumps(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        ) + "\n\n"

    async def gen():
        yield frame({"role": "assistant"})
        words = content.split(" ")
        for i in range(0, len(words), 16):
            yield frame({"content": " ".join(words[i : i + 16]) + " "})
        yield frame({}, finish="stop")
        yield "data: [DONE]\n\n"

    return gen


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model = req.model or settings.openai_model
    hits = rag.INDEX.query(_last_user(req.messages), top_k=settings.rag_top_k)
    system_context, citations = _context_block(hits)

    if settings.openai_api_key:
        return await _proxy_openai(req, model, system_context, citations)

    answer = _offline_answer(hits, citations)
    if req.stream:
        return StreamingResponse(_offline_stream(model, answer)(), media_type="text/event-stream")
    return JSONResponse(_completion_obj(model, answer, citations))


async def _proxy_openai(req: ChatCompletionRequest, model: str, system_context: str, citations: list[str]):
    payload = req.model_dump(exclude_none=True)
    payload["model"] = model
    payload["messages"] = [{"role": "system", "content": system_context}] + [
        {"role": m.role, "content": m.content} for m in req.messages
    ]
    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    if req.stream:
        async def gen():
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    async for raw in resp.aiter_bytes():
                        yield raw

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers={"x-citations": ",".join(citations)}
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(url, json=payload, headers=headers)
    try:
        data = resp.json()
    except ValueError:
        return JSONResponse({"error": "upstream returned non-JSON", "body": resp.text[:2000]}, status_code=502)
    if isinstance(data, dict):
        data["x_citations"] = citations
    return JSONResponse(data, status_code=resp.status_code)
