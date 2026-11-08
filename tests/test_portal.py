"""End-to-end tests: the real Anthropic SDK is used, with the HTTP layer mocked,
so request shapes and stream parsing are exercised without an API key."""
import json
import os
import tempfile

os.environ["DATA_DIR"] = tempfile.mkdtemp()
os.environ["ADMIN_PASSWORD"] = "secret"

import anthropic  # noqa: E402
import httpx2  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from portal import claude_service, db, main  # noqa: E402

REQUESTS: list[dict] = []
NEXT_TEXT: list[str] = []


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def message_stream(text: str, stop_reason="end_turn"):
    usage = {"input_tokens": 10, "output_tokens": 5}
    return sse([
        {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-opus-5-5", "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig123"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        *[{"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": chunk}}
          for chunk in [text[: len(text) // 2], text[len(text) // 2:]]],
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])


def handler(request: httpx2.Request) -> httpx2.Response:
    path = request.url.path
    if path == "/v1/files" and request.method == "POST":
        n = sum(1 for r in REQUESTS if r["path"] == "/v1/files")
        REQUESTS.append({"path": path})
        return httpx2.Response(200, json={
            "id": f"file_{n}", "type": "file", "filename": "x", "mime_type": "application/pdf",
            "size_bytes": 1, "created_at": "2026-11-08T00:00:00Z", "downloadable": False})
    if path.startswith("/v1/files/") and request.method == "DELETE":
        REQUESTS.append({"path": path})
        return httpx2.Response(200, json={"id": path.rsplit("/", 1)[1], "type": "file_deleted"})
    if path == "/v1/messages":
        body = json.loads(request.content)
        REQUESTS.append({"path": path, "body": body, "headers": dict(request.headers)})
        text = NEXT_TEXT.pop(0) if NEXT_TEXT else "Hello from the coach."
        stop = "refusal" if text == "<refuse>" else "end_turn"
        return httpx2.Response(200, text=message_stream("" if stop == "refusal" else text, stop),
                               headers={"content-type": "text/event-stream"})
    return httpx2.Response(404, json={"error": {"type": "not_found_error", "message": path}})


@pytest.fixture(scope="module")
def app_client():
    main._api = anthropic.AsyncAnthropic(
        api_key="test", http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    with TestClient(main.app) as tc:
        yield tc


@pytest.fixture(scope="module")
def portal(app_client):
    r = app_client.post("/api/admin/clients", json={"name": "Alex", "notes": "Prefers Chinese. Test in Dec."},
                        headers={"X-Admin-Password": "secret"})
    assert r.status_code == 200
    return {"X-Client-Token": r.json()["token"]}


def chat(app_client, h, conv, text):
    r = app_client.post("/api/chat", json={"conversation_id": conv, "message": text}, headers=h)
    assert r.status_code == 200
    return [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]


def test_admin_auth(app_client):
    assert app_client.get("/api/admin/clients").status_code == 401
    assert app_client.get("/api/admin/clients", headers={"X-Admin-Password": "nope"}).status_code == 401
    assert app_client.get("/api/me", headers={"X-Client-Token": "bad"}).status_code == 401


def test_upload_rejects_bad_type(app_client, portal):
    r = app_client.post("/api/files", files={"file": ("a.exe", b"MZ", "application/octet-stream")}, headers=portal)
    assert r.status_code == 400


def test_chat_flow_is_append_only(app_client, portal):
    r = app_client.post("/api/files", files={"file": ("report.pdf", b"%PDF-1.4 test", "application/pdf")}, headers=portal)
    assert r.status_code == 200, r.text
    conv = app_client.get("/api/chat", headers=portal).json()["conversation_id"]

    events = chat(app_client, portal, conv, "What does my report say?")
    assert "".join(e.get("text", "") for e in events if e["type"] == "text") == "Hello from the coach."
    assert events[-1]["type"] == "done"

    first = REQUESTS[-1]
    assert first["body"]["model"] == "claude-opus-5-5"
    assert first["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in first["headers"]["anthropic-beta"]
    blocks = first["body"]["messages"][0]["content"]
    assert blocks[0]["text"].startswith("[Portal context")
    assert {"type": "document", "source": {"type": "file", "file_id": "file_0"}, "title": "report.pdf"} in blocks

    # second turn: a new image upload is attached to the new turn only
    app_client.post("/api/files", files={"file": ("q.png", b"\x89PNG", "image/png")}, headers=portal)
    chat(app_client, portal, conv, "And this screenshot?")
    second = REQUESTS[-1]["body"]["messages"]
    assert second[:2] == [first["body"]["messages"][0], second[1]]  # first turn unchanged
    assert second[0] == first["body"]["messages"][0]
    assert second[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig123"}  # replayed unchanged
    new_turn = second[2]["content"]
    assert not any(b.get("text", "").startswith("[Portal context") for b in new_turn)  # context unchanged
    assert {"type": "image", "source": {"type": "file", "file_id": "file_1"}} in new_turn
    assert not any(b.get("source", {}).get("file_id") == "file_0" for b in new_turn)

    shown = app_client.get("/api/chat", headers=portal).json()["messages"]
    assert [m["role"] for m in shown] == ["user", "assistant", "user", "assistant"]
    assert shown[0]["text"] == "What does my report say?"


def test_refusal_rolls_back_turn(app_client, portal):
    conv = app_client.get("/api/chat", headers=portal).json()["conversation_id"]
    before = len(app_client.get("/api/chat", headers=portal).json()["messages"])
    NEXT_TEXT.append("<refuse>")
    events = chat(app_client, portal, conv, "something")
    assert events[-1]["type"] == "error"
    assert len(app_client.get("/api/chat", headers=portal).json()["messages"]) == before


def test_practice_generate_hide_answers_and_grade(app_client, portal):
    pset = {
        "title": "Number series", "rationale": "Targets your weaker numerical index.",
        "questions": [
            {"area": "numerical", "difficulty": 2, "prompt": "2, 4, 8, ?", "options": ["10", "16", "12", "14"],
             "correct_index": 1, "explanation": "Doubling."},
            {"area": "verbal", "difficulty": 3, "prompt": "Hot : cold :: up : ?", "options": ["down", "left", "high", "over"],
             "correct_index": 0, "explanation": "Antonyms."},
            {"area": "logic", "difficulty": 3, "prompt": "bad key", "options": ["a", "b"],
             "correct_index": 7, "explanation": "invalid, should be dropped"},
        ],
    }
    NEXT_TEXT.append(json.dumps(pset))
    r = app_client.post("/api/practice", json={"area": "mixed", "difficulty": "adaptive", "count": 5}, headers=portal)
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["total"] == 2
    assert all("correct_index" not in q and "explanation" not in q for q in s["questions"])
    body = REQUESTS[-1]["body"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "high"

    assert "correct_index" not in json.dumps(app_client.get(f"/api/practice/{s['id']}", headers=portal).json())

    graded = app_client.post(f"/api/practice/{s['id']}/submit", json={"answers": [1, 2]}, headers=portal).json()
    assert graded["score"] == 1 and graded["questions"][0]["correct_index"] == 1
    assert app_client.post(f"/api/practice/{s['id']}/submit", json={"answers": [1, 0]}, headers=portal).status_code == 400

    client = db.client_by_token(portal["X-Client-Token"])
    summary = claude_service.progress_summary(client["id"])
    assert "numerical: 1/1 (100%)" in summary and "verbal: 0/1 (0%)" in summary


def test_other_client_cannot_see_data(app_client, portal):
    other = app_client.post("/api/admin/clients", json={"name": "Sam"}, headers={"X-Admin-Password": "secret"}).json()
    h = {"X-Client-Token": other["token"]}
    assert app_client.get("/api/files", headers=h).json() == []
    assert app_client.get("/api/practice", headers=h).json() == []
    mine = app_client.get("/api/files", headers=portal).json()[0]["id"]
    assert app_client.get(f"/api/files/{mine}/raw", headers=h).status_code == 404
    conv = app_client.get("/api/chat", headers=portal).json()["conversation_id"]
    assert app_client.post("/api/chat", json={"conversation_id": conv, "message": "hi"}, headers=h).status_code == 404


def test_delete_client_cleans_up(app_client):
    c = app_client.post("/api/admin/clients", json={"name": "Temp"}, headers={"X-Admin-Password": "secret"}).json()
    h = {"X-Client-Token": c["token"]}
    app_client.post("/api/files", files={"file": ("n.txt", b"notes", "text/plain")}, headers=h)
    r = app_client.delete(f"/api/admin/clients/{c['id']}", headers={"X-Admin-Password": "secret"})
    assert r.status_code == 200
    assert REQUESTS[-1]["path"].startswith("/v1/files/")
    assert app_client.get("/api/me", headers=h).status_code == 401
