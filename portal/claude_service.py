"""Everything that talks to the Claude API: file upload, coaching chat, practice-set generation."""
import json
import os
from typing import AsyncIterator

import anthropic
from pydantic import BaseModel, Field

from . import db

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
# Server-side refusal fallback: if the model declines a request, the API
# re-runs it on a suitable fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
DOC_MIMES = {"application/pdf", "text/plain", "text/markdown"}
ALLOWED_MIMES = IMAGE_MIMES | DOC_MIMES

AREAS = {
    "verbal": "Verbal reasoning (analogies, vocabulary in context, odd-one-out, verbal classification)",
    "numerical": "Numerical reasoning (number series, arithmetic word problems, quantitative comparisons)",
    "pattern": "Abstract pattern reasoning (letter/number/symbol sequences and matrices described in text)",
    "logic": "Logical deduction (syllogisms, ordering/ranking puzzles, conditional reasoning)",
    "spatial": "Spatial reasoning described in words (rotations, folding, directions, cube nets)",
    "memory": "Working-memory style tasks (mental manipulation of sequences, digit/letter reordering)",
    "mixed": "A balanced mix of all areas, weighted toward the client's weaker areas",
}

SYSTEM_PROMPT = """You are the coaching assistant inside a private client portal run by an IQ-test preparation coach. \
Each conversation is with one client of the coach.

What you help with:
- Explaining the client's uploaded documents (for example an IQ/cognitive assessment report, score sheets, \
or screenshots of practice questions) in plain language: what each index or subtest measures, what their \
relative strengths and weaker areas look like, and what that means for practice.
- Explaining how to approach specific question types, walking through worked solutions step by step, and \
giving short practice questions in the chat when asked.
- Building a sensible practice plan around the client's weaker areas.

Ground rules:
- Reply in the language the client writes in.
- When you refer to the client's documents, be specific about what they show; if something is unclear or \
not in the documents, say so rather than guessing.
- Do not estimate or announce an IQ score, diagnose any condition, or present practice results as a real \
test score. Practice performance only shows progress on practice material.
- Do not reproduce items from published, copyrighted tests (e.g. WAIS, WISC, Stanford-Binet, Raven's). \
Write original questions in the same style instead.
- Be encouraging and concrete. Keep answers focused; use short headings or lists when they help.
- For a scored practice set that is saved to their progress, point the client to the "Practice" tab of \
the portal, which generates customized sets from their documents and past results."""


def get_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


# ---------------------------------------------------------------- files

def file_block(f: dict) -> dict:
    source = {"type": "file", "file_id": f["anthropic_file_id"]}
    if f["mime"] in IMAGE_MIMES:
        return {"type": "image", "source": source}
    return {"type": "document", "source": source, "title": f["filename"]}


async def upload_to_claude(client: anthropic.AsyncAnthropic, filename: str, data: bytes, mime: str) -> str:
    # Markdown is sent as plain text: document blocks accept text/plain.
    upload_mime = "text/plain" if mime == "text/markdown" else mime
    uploaded = await client.files.upload(file=(filename, data, upload_mime))
    return uploaded.id


async def delete_from_claude(client: anthropic.AsyncAnthropic, anthropic_file_id: str) -> None:
    try:
        await client.files.delete(anthropic_file_id)
    except anthropic.NotFoundError:
        pass


# ---------------------------------------------------------------- progress summary

def progress_summary(client_id: int) -> str:
    """Per-area accuracy across submitted practice sets, as plain text."""
    per_area: dict[str, list[int]] = {}
    recent_misses: list[str] = []
    for s in db.list_practice_sets(client_id):
        if s["submitted_at"] is None:
            continue
        questions = json.loads(s["questions_json"])
        answers = json.loads(s["answers_json"] or "[]")
        for i, q in enumerate(questions):
            given = answers[i] if i < len(answers) else None
            ok = given == q["correct_index"]
            stats = per_area.setdefault(q["area"], [0, 0])
            stats[0] += int(ok)
            stats[1] += 1
            if not ok and len(recent_misses) < 8:
                recent_misses.append(f"- [{q['area']}, difficulty {q['difficulty']}] {q['prompt'][:160]}")
    if not per_area:
        return "No practice sets completed yet."
    lines = ["Practice accuracy by area (all completed sets):"]
    for area, (right, total) in sorted(per_area.items()):
        lines.append(f"- {area}: {right}/{total} ({round(100 * right / total)}%)")
    if recent_misses:
        lines.append("Most recent questions answered incorrectly:")
        lines.extend(recent_misses)
    return "\n".join(lines)


def portal_context(client: dict) -> str:
    parts = [f"[Portal context — not written by the client]\nClient name: {client['name']}"]
    if client["notes"].strip():
        parts.append(f"Coach's notes about this client:\n{client['notes'].strip()}")
    parts.append(progress_summary(client["id"]))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- chat

def _history(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": json.loads(m["content_json"])} for m in messages]


def _attached_file_ids(history: list[dict]) -> set[str]:
    ids = set()
    for m in history:
        if m["role"] != "user":
            continue
        for block in m["content"]:
            src = block.get("source") if isinstance(block, dict) else None
            if src and src.get("type") == "file":
                ids.add(src["file_id"])
    return ids


def _last_context(history: list[dict]) -> str | None:
    for m in reversed(history):
        if m["role"] == "user":
            for block in m["content"]:
                if block.get("type") == "text" and block["text"].startswith("[Portal context"):
                    return block["text"]
    return None


def build_user_turn(client: dict, history: list[dict], text: str) -> list[dict]:
    """New user turn. The history is only ever appended to (never edited), so earlier
    turns, their cache and their thinking blocks stay valid. New uploads and changed
    progress data are therefore attached to the turn where they first appear."""
    content: list[dict] = []
    context = portal_context(client)
    if context != _last_context(history):
        content.append({"type": "text", "text": context})

    attached = _attached_file_ids(history)
    new_files = [f for f in db.list_files(client["id"]) if f["anthropic_file_id"] not in attached]
    if new_files:
        names = ", ".join(f["filename"] for f in new_files)
        content.append({"type": "text", "text": f"[Documents uploaded by the client to the portal: {names}]"})
        content.extend(file_block(f) for f in new_files)

    content.append({"type": "text", "text": text})
    return content


def _storable_content(blocks) -> list[dict]:
    content = [b.model_dump(mode="json", exclude_none=True) for b in blocks]
    # After a mid-output fallback, blocks before the last fallback marker that the
    # fallback model can't continue from (thinking, tool use) must not be echoed back.
    fallback_idx = [i for i, b in enumerate(content) if b.get("type") == "fallback"]
    if fallback_idx:
        last = fallback_idx[-1]
        content = [
            b for i, b in enumerate(content)
            if i > last or b.get("type") not in {"thinking", "redacted_thinking", "tool_use", "fallback"}
        ]
    return content


def _visible_text(content: list[dict]) -> str:
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


async def stream_chat(
    api: anthropic.AsyncAnthropic, client: dict, conversation_id: int, text: str
) -> AsyncIterator[dict]:
    """Yields {"type": "text"|"done"|"error", ...} events and persists the turn."""
    history = _history(db.get_messages(conversation_id))
    user_content = build_user_turn(client, history, text)
    user_msg_id = db.add_message(conversation_id, "user", user_content, text)
    messages = history + [{"role": "user", "content": user_content}]

    try:
        async with api.beta.messages.stream(
            model=MODEL,
            max_tokens=32000,
            system=SYSTEM_PROMPT,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    yield {"type": "text", "text": event.text}
            final = await stream.get_final_message()
    except anthropic.APIError as e:
        db.delete_message(user_msg_id)  # drop the unanswered turn from the tail
        yield {"type": "error", "message": f"Claude API error: {getattr(e, 'message', str(e))}"}
        return

    if final.stop_reason == "refusal":
        db.delete_message(user_msg_id)
        yield {"type": "error", "message": "Sorry, I can't help with that request."}
        return

    content = _storable_content(final.content)
    db.add_message(conversation_id, "assistant", content, _visible_text(content))
    yield {"type": "done", "truncated": final.stop_reason == "max_tokens"}


# ---------------------------------------------------------------- practice sets

class Question(BaseModel):
    area: str = Field(description="One of: verbal, numerical, pattern, logic, spatial, memory")
    difficulty: int = Field(description="1 (easy) to 5 (very hard)")
    prompt: str = Field(description="The full question text, self-contained, solvable from text alone")
    options: list[str] = Field(description="4 or 5 answer options, without letter prefixes")
    correct_index: int = Field(description="0-based index into options of the single correct answer")
    explanation: str = Field(description="Step-by-step worked solution and the general strategy it practices")


class PracticeSet(BaseModel):
    title: str
    rationale: str = Field(
        description="2-4 sentences to the client on why this set targets what it does, based on their documents and past results"
    )
    questions: list[Question]


DIFFICULTIES = {"adaptive", "easy", "medium", "hard"}


async def generate_practice_set(
    api: anthropic.AsyncAnthropic, client: dict, area: str, difficulty: str, count: int, focus: str
) -> PracticeSet:
    files = db.list_files(client["id"])
    content: list[dict] = []
    if files:
        content.append({"type": "text", "text": "Documents the client uploaded:"})
        content.extend(file_block(f) for f in files)
    diff_text = (
        "Choose difficulty per question from the client's past accuracy: aim for roughly 70% expected accuracy, "
        "easier in areas where they struggle, harder where they are strong."
        if difficulty == "adaptive"
        else f"Overall difficulty: {difficulty}."
    )
    content.append({
        "type": "text",
        "text": f"""{portal_context(client)}

Create a practice set of exactly {count} original multiple-choice questions for this client.
Area: {AREAS[area]}
{diff_text}
{f"The client asked to focus on: {focus}" if focus.strip() else ""}

Requirements:
- Target the weaker areas and question types shown in the documents and past results; don't repeat \
questions they have already seen.
- Every question must be fully solvable from its text (no images), have exactly one correct answer, \
and 4-5 plausible options. Double-check each answer key and that no other option is also defensible.
- Write the title, rationale, questions, options and explanations in the client's language \
(infer it from their documents and the coach's notes; default to English).""",
    })

    # Streamed (long thinking + long output would risk an HTTP timeout otherwise);
    # get_final_message() returns the parsed, schema-validated result.
    async with api.beta.messages.stream(
        model=MODEL,
        max_tokens=64000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        output_format=PracticeSet,
        betas=[FALLBACK_BETA],
        fallbacks="default",
    ) as stream:
        response = await stream.get_final_message()
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise ValueError("The model could not generate this practice set. Try a different focus.")

    result = response.parsed_output
    result.questions = [
        q for q in result.questions
        if 2 <= len(q.options) <= 6 and 0 <= q.correct_index < len(q.options)
    ]
    if not result.questions:
        raise ValueError("No valid questions were generated. Please try again.")
    return result
