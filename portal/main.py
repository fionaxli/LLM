"""FastAPI app: client portal (/p/<token>) and coach admin page (/admin)."""
import json
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import claude_service as cs
from . import db

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init()
    yield


app = FastAPI(title="IQ Coaching Portal", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_api: anthropic.AsyncAnthropic | None = None


def api() -> anthropic.AsyncAnthropic:
    global _api
    if _api is None:
        _api = cs.get_client()
    return _api


# ---------------------------------------------------------------- auth

def current_client(x_client_token: str = Header(default="")) -> dict:
    client = db.client_by_token(x_client_token) if x_client_token else None
    if not client:
        raise HTTPException(401, "Invalid or expired portal link.")
    return client


def require_admin(x_admin_password: str = Header(default="")) -> None:
    expected = os.environ.get("ADMIN_PASSWORD", "")
    if not expected or not secrets.compare_digest(x_admin_password, expected):
        raise HTTPException(401, "Wrong admin password.")


# ---------------------------------------------------------------- pages

@app.get("/")
def home():
    return FileResponse(STATIC / "home.html")


@app.get("/p/{token}")
def portal_page(token: str):
    return FileResponse(STATIC / "portal.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC / "admin.html")


# ---------------------------------------------------------------- client: profile & files

@app.get("/api/me")
def me(client: dict = Depends(current_client)):
    return {"name": client["name"]}


def _file_out(f: dict) -> dict:
    return {k: f[k] for k in ("id", "filename", "mime", "size", "created_at")}


@app.get("/api/files")
def list_files(client: dict = Depends(current_client)):
    return [_file_out(f) for f in db.list_files(client["id"])]


@app.post("/api/files")
async def upload_file(file: UploadFile = File(...), client: dict = Depends(current_client)):
    mime = (file.content_type or "").split(";")[0].strip().lower()
    name = Path(file.filename or "upload").name
    if name.lower().endswith(".md"):
        mime = "text/markdown"
    if mime not in cs.ALLOWED_MIMES:
        raise HTTPException(400, "Supported files: PDF, PNG, JPG, GIF, WEBP, TXT, MD.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Files must be under 30 MB.")

    try:
        anthropic_id = await cs.upload_to_claude(api(), name, data, mime)
    except anthropic.APIError as e:
        raise HTTPException(502, f"Could not process the file: {getattr(e, 'message', str(e))}")

    client_dir = db.UPLOAD_DIR / str(client["id"])
    client_dir.mkdir(parents=True, exist_ok=True)
    path = client_dir / f"{uuid.uuid4().hex}{Path(name).suffix}"
    path.write_bytes(data)
    return _file_out(db.add_file(client["id"], name, mime, len(data), str(path), anthropic_id))


@app.get("/api/files/{file_id}/raw")
def download_file(file_id: int, client: dict = Depends(current_client)):
    f = db.get_file(client["id"], file_id)
    if not f:
        raise HTTPException(404)
    return FileResponse(f["path"], media_type=f["mime"], filename=f["filename"])


@app.delete("/api/files/{file_id}")
def remove_file(file_id: int, client: dict = Depends(current_client)):
    if not db.get_file(client["id"], file_id):
        raise HTTPException(404)
    db.hide_file(client["id"], file_id)
    return {"ok": True}


# ---------------------------------------------------------------- client: chat

@app.get("/api/chat")
def get_chat(client: dict = Depends(current_client)):
    conv = db.latest_conversation(client["id"])
    return {
        "conversation_id": conv["id"],
        "messages": [
            {"role": m["role"], "text": m["display_text"]} for m in db.get_messages(conv["id"])
        ],
    }


@app.post("/api/chat/new")
def new_chat(client: dict = Depends(current_client)):
    return {"conversation_id": db.new_conversation(client["id"])["id"], "messages": []}


class ChatIn(BaseModel):
    conversation_id: int
    message: str = Field(min_length=1, max_length=8000)


@app.post("/api/chat")
async def chat(body: ChatIn, client: dict = Depends(current_client)):
    if not db.get_conversation(client["id"], body.conversation_id):
        raise HTTPException(404, "Conversation not found.")

    async def events():
        async for ev in cs.stream_chat(api(), client, body.conversation_id, body.message.strip()):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- client: practice

class PracticeIn(BaseModel):
    area: str = "mixed"
    difficulty: str = "adaptive"
    count: int = Field(default=10, ge=3, le=20)
    focus: str = Field(default="", max_length=500)


def _set_out(s: dict, reveal: bool) -> dict:
    questions = json.loads(s["questions_json"])
    out = {
        "id": s["id"], "title": s["title"], "rationale": s["rationale"], "area": s["area"],
        "difficulty": s["difficulty"], "total": s["total"], "score": s["score"],
        "created_at": s["created_at"], "submitted": s["submitted_at"] is not None,
    }
    if reveal:
        out["answers"] = json.loads(s["answers_json"] or "[]")
        out["questions"] = questions
    else:  # never send answer keys before the client submits
        out["questions"] = [
            {k: q[k] for k in ("area", "difficulty", "prompt", "options")} for q in questions
        ]
    return out


@app.post("/api/practice")
async def create_practice(body: PracticeIn, client: dict = Depends(current_client)):
    if body.area not in cs.AREAS or body.difficulty not in cs.DIFFICULTIES:
        raise HTTPException(400, "Unknown area or difficulty.")
    try:
        result = await cs.generate_practice_set(api(), client, body.area, body.difficulty, body.count, body.focus)
    except ValueError as e:
        raise HTTPException(502, str(e))
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API error: {getattr(e, 'message', str(e))}")
    set_id = db.add_practice_set(
        client["id"], body.area, body.difficulty, result.title, result.rationale,
        [q.model_dump() for q in result.questions],
    )
    return _set_out(db.get_practice_set(client["id"], set_id), reveal=False)


@app.get("/api/practice")
def list_practice(client: dict = Depends(current_client)):
    return [
        {k: v for k, v in _set_out(s, reveal=False).items() if k != "questions"}
        for s in db.list_practice_sets(client["id"])
    ]


@app.get("/api/practice/{set_id}")
def get_practice(set_id: int, client: dict = Depends(current_client)):
    s = db.get_practice_set(client["id"], set_id)
    if not s:
        raise HTTPException(404)
    return _set_out(s, reveal=s["submitted_at"] is not None)


class SubmitIn(BaseModel):
    answers: list[int | None]


@app.post("/api/practice/{set_id}/submit")
def submit_practice(set_id: int, body: SubmitIn, client: dict = Depends(current_client)):
    s = db.get_practice_set(client["id"], set_id)
    if not s:
        raise HTTPException(404)
    if s["submitted_at"] is not None:
        raise HTTPException(400, "Already submitted.")
    questions = json.loads(s["questions_json"])
    answers = (body.answers + [None] * len(questions))[: len(questions)]
    score = sum(1 for q, a in zip(questions, answers) if a == q["correct_index"])
    db.submit_practice_set(set_id, answers, score)
    return _set_out(db.get_practice_set(client["id"], set_id), reveal=True)


# ---------------------------------------------------------------- admin

class ClientIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    notes: str = Field(default="", max_length=5000)


class ClientPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=5000)


@app.get("/api/admin/clients", dependencies=[Depends(require_admin)])
def admin_list_clients():
    return db.list_clients()


@app.post("/api/admin/clients", dependencies=[Depends(require_admin)])
def admin_create_client(body: ClientIn):
    return db.create_client(body.name.strip(), body.notes)


@app.patch("/api/admin/clients/{client_id}", dependencies=[Depends(require_admin)])
def admin_update_client(client_id: int, body: ClientPatch):
    if not db.get_client(client_id):
        raise HTTPException(404)
    db.update_client(client_id, body.name, body.notes)
    return db.get_client(client_id)


@app.get("/api/admin/clients/{client_id}", dependencies=[Depends(require_admin)])
def admin_client_detail(client_id: int):
    client = db.get_client(client_id)
    if not client:
        raise HTTPException(404)
    return {
        "client": client,
        "files": [_file_out(f) for f in db.list_files(client_id)],
        "practice_sets": [
            {k: v for k, v in _set_out(s, reveal=False).items() if k != "questions"}
            for s in db.list_practice_sets(client_id)
        ],
        "progress": cs.progress_summary(client_id),
    }


@app.get("/api/admin/clients/{client_id}/files/{file_id}/raw", dependencies=[Depends(require_admin)])
def admin_download_file(client_id: int, file_id: int):
    f = db.get_file(client_id, file_id)
    if not f:
        raise HTTPException(404)
    return FileResponse(f["path"], media_type=f["mime"], filename=f["filename"])


@app.delete("/api/admin/clients/{client_id}", dependencies=[Depends(require_admin)])
async def admin_delete_client(client_id: int):
    files = db.delete_client(client_id)
    for f in files:
        Path(f["path"]).unlink(missing_ok=True)
        if f["anthropic_file_id"]:
            await cs.delete_from_claude(api(), f["anthropic_file_id"])
    return {"ok": True}
