# IQ Coaching Client Portal

A private portal for each coaching client. Each client gets their own link where they can:

- **Upload documents**: an IQ/cognitive assessment report (PDF), score sheets, screenshots of practice questions (PNG/JPG/WEBP/GIF), or notes (TXT/MD).
- **Chat with a coaching bot** (powered by Claude). It has read their documents and can explain their results, walk through question types and solutions, and suggest a practice plan. Replies stream in as they're written.
- **Generate customized practice sets**: multiple-choice questions aimed at their weaker areas, based on their documents, your notes and their past practice results. Difficulty adapts to how they've been doing. Answer keys stay on the server until the client submits; then they get a score and a worked explanation for each question.

The interface is in English and Chinese (toggle in the top right). The bot replies in whatever language the client writes in.

As the coach, you get an **admin page** (`/admin`). There you create a portal for each client, write private notes the bot will use (goals, test date, observations), view their uploads and see their practice accuracy by area.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in ANTHROPIC_API_KEY and ADMIN_PASSWORD
set -a; source .env; set +a
uvicorn portal.main:app --host 0.0.0.0 --port 8000
```

1. Open `http://localhost:8000/admin` and log in with `ADMIN_PASSWORD`.
2. Create the client, add notes, and click **Copy link**.
3. Send the link (`/p/<private-token>`) to the client.

### Docker

```bash
docker build -t iq-portal .
docker run -p 8000:8000 --env-file .env -v iq-portal-data:/data iq-portal
```

## Configuration

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key |
| `ADMIN_PASSWORD` | Password for `/admin`. The admin page stays locked if this isn't set. |
| `DATA_DIR` | Where the SQLite database and uploaded files are stored (default `./data`) |
| `CLAUDE_MODEL` | Optional model override (default `claude-opus-5-5`) |

## Before you put it online

- **Use HTTPS.** Clients upload personal assessment reports, so run the portal behind a TLS-terminating host or proxy (Caddy, nginx, Render, Fly.io, etc.).
- **The link is the login.** Anyone who has a client's link can open that portal, so send it privately. To revoke access, delete the client on the admin page. That also deletes their files from your server and from Anthropic.
- **Back up `DATA_DIR`.** It holds the database and the original uploaded files.
- **Uploaded files go to Anthropic.** They are sent through the Claude Files API so the bot can read them. Mention this in your privacy notice or client agreement.

## How it works

```
portal/
  main.py            FastAPI routes: client API (X-Client-Token), admin API (X-Admin-Password), pages
  claude_service.py  Claude calls: file upload, streaming chat, practice-set generation
  db.py              SQLite schema and queries
  static/            portal.html (client), admin.html (coach), style.css
tests/test_portal.py End-to-end tests using the real Anthropic SDK with mocked HTTP
```

- **Documents** are uploaded once to the Claude Files API and referenced by `file_id`. They aren't re-sent as base64 on every message.
- **Chat history is append-only.** A document uploaded mid-conversation, or a change in the client's practice results, is attached to the next message rather than inserted into earlier ones. This keeps prompt caching effective and keeps the model's earlier reasoning valid. Use **New chat** for a clean start. Deleting a document hides it from new chats and new practice sets; past chats that already used it keep working.
- **Practice sets** come back as schema-validated JSON (structured outputs). Questions with an invalid answer key are dropped. Grading happens on the server.
- **Refusal fallback** is turned on (`fallbacks: "default"`). If the model declines a request, the API retries it on a fallback model within the same call. If the request is still refused, the turn is rolled back and the client sees a short message.
- The bot is told not to estimate IQ scores or diagnose anything, and to write original questions rather than copy items from published tests (WAIS, Raven's, etc.).

## Tests

```bash
pip install pytest
python -m pytest -q
```

The tests need no API key. They cover access control between clients, file upload, the streamed chat flow (including checks that history is never rewritten), refusal rollback, hiding answer keys before submission, grading, the progress summary, and cleanup when a client is deleted.
