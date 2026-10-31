# RAG Chatbot for Persian and English Documents

A chatbot that answers questions using only the documents you upload. You upload PDF, TXT or Markdown files, ask questions in Persian or English, and get answers with numbered citations that point to the file and page they came from. Each user sees and searches only their own files.

The whole stack runs in Docker Compose. Nothing needs to be installed on the host except Docker.

## Features

- **Answers from your files only.** If the documents do not contain the answer, the bot says so in the language of the question instead of guessing.
- **Numbered citations.** Every claim is followed by a citation like `[2]`. Clicking it shows the source text, file name, page and relevance score.
- **Persian and English.** You can ask in either language. A question in English can be answered from a Persian document.
- **Follow-up questions.** Questions like "when did it start?" are rewritten into standalone questions using the last three turns of the conversation. The rewritten search query is shown under the question.
- **Several documents at once.** When more than one file answers a question, the answer covers each of them, and no single file can take every source slot.
- **Streaming answers.** Answers appear word by word over server-sent events.
- **Background processing with live progress.** Uploads return right away. A worker embeds the file in the background, and the documents panel shows `queued`, then `embedding 0…99%`, then `ready`.
- **Accounts and saved conversations.** Email and password sign-in. Conversations are saved and can be reopened or deleted.
- **Light and dark theme.** The interface follows the system setting.

## Architecture

```
Browser ──> Next.js (frontend, :3000) ──/api/*──> FastAPI (api, :8000) ──> Postgres + pgvector (db)
                                                       │
                                                       └─ queue ──> RabbitMQ ──> Celery worker ──> Gemini embeddings
```

| Service | Image or build | Role |
|---|---|---|
| `db` | `pgvector/pgvector:0.8.7-pg18-trixie` | Documents, chunks, vectors, users and conversations. Not exposed to the host. |
| `rabbitmq` | `rabbitmq:4.4.0-management-alpine` | Job queue for embedding. Management panel on `127.0.0.1:15672`. |
| `worker` | `./backend` | Celery worker that embeds uploaded documents. |
| `api` | `./backend` | FastAPI app on `127.0.0.1:8000`. |
| `frontend` | `./frontend` | Next.js app on `127.0.0.1:3000`. |
| `app` | `./backend` | One-off container for tests and command-line tools. |

The browser only talks to the Next.js server. Next.js forwards `/api/*` to FastAPI, so the API needs no CORS setup and the session cookie stays same-origin.

## Tech stack

| Area | Choice |
|---|---|
| Chat model | Google `gemini-3.5-flash-lite` with minimal thinking |
| Embeddings | Google `gemini-embedding-001`, 768 dimensions |
| Reranker | Cohere `rerank-v3.5` (optional) |
| Vector search | pgvector with an HNSW index and cosine distance |
| Keyword search | BM25 written in SQL over a Postgres `tsvector` column |
| Backend | Python 3.12, FastAPI 0.149, Uvicorn, psycopg 3 |
| PDF parsing | PyMuPDF |
| Background jobs | Celery 5.6 with RabbitMQ 4.4 |
| Auth | JWT (PyJWT) in an httpOnly cookie, passwords hashed with scrypt from the standard library |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| Tests | pytest against a real Postgres, Node's built-in test runner for the frontend |

No RAG framework is used. Each step of the pipeline is written directly, so every decision is visible in the code and can be measured on its own.

## How it works

### Ingestion: from upload to searchable chunks

1. **Validate.** The API checks the extension, the 20 MB limit, and for PDFs that the file really starts with `%PDF-`. A SHA-256 hash stops the same user from uploading the same file twice.
2. **Extract text.** PyMuPDF reads each page. Persian PDFs need several fixes before the text is usable:
   - The Lam-Alef ligature is often stored as two reversed letters plus a zero-width character. These are detected by glyph width and put back in the right order.
   - Headers and footers that repeat on many pages are removed. Only lines at the edge of the page with at least 10 letters and no digits are removed, so page numbers and table data are kept.
   - Table cells that PyMuPDF returns as separate lines are joined back into one row by their vertical position, so a label stays next to its numbers (`Cash | 838,212 | 539,008`).
3. **Normalize.** Arabic Yeh and Kaf become Persian ones, Persian and Arabic-Indic digits become ASCII digits, Persian thousands and decimal separators become `,` and `.`, and invisible characters are removed. The zero-width non-joiner is kept because it is part of Persian spelling.
4. **Chunk.** Text is split into chunks of about 800 characters with 150 characters of overlap. Chunks never cross a page boundary, so every chunk has exactly one page number to cite.
5. **Embed in the background.** The API saves the document as `processing` and puts a job on RabbitMQ. The Celery worker embeds chunks in batches of 10 and updates progress after each batch.

The worker is built to survive failures:

- Messages are acknowledged only after the work is done (`task_acks_late`), so a crashed worker's job is delivered again.
- A redelivered job for a document that is no longer `processing` is skipped, and chunks are saved in one transaction, so a retry never creates duplicate chunks. This was tested by killing the worker in the middle of a job.
- Rate limits (429) and server errors (5xx) are retried three times with backoff (60, 120, 240 seconds). Other errors mark the document as `failed` with a readable message.

### Answering a question

1. **Rewrite follow-ups.** If the conversation has history, Gemini rewrites the question into a standalone one. If the rewrite fails or changes nothing, the typed question is used.
2. **Normalize the question** the same way documents were normalized, so `۲,۰۵۲,۲۳۶` in a question matches `2,052,236` in a file.
3. **Search two ways, 15 candidates each.**
   - **Vector search** finds chunks with the same meaning, even with different words or a different language.
   - **BM25 keyword search** finds exact numbers, note codes and rare names, which vector search often misses.
4. **Fuse with Reciprocal Rank Fusion (RRF).** The two score scales cannot be compared, so RRF combines the lists by rank: each chunk gets `1 / (k + rank)` from each list.
5. **Rerank.** The top 20 fused candidates go to Cohere's cross-encoder, which reads the question and each chunk together. If Cohere is slow or fails, the fused order is used and the answer still goes out.
6. **Spread across files.** Five chunks are picked, at most four from one file, so a second document that also answers the question still reaches the model.
7. **Generate.** The chunks are sent inside `<source>` tags with numbers. The prompt asks for citations after every claim, numbers copied exactly, and an answer in the question's language. The answer streams to the browser, and citation numbers outside the valid range are dropped.

### Search details

**BM25 in SQL.** Postgres's built-in `ts_rank` has no inverse document frequency, so a common word like «به» counted as much as a rare number. The project computes real BM25 in SQL instead, using statistics from the current user's chunks only. Other users' files never affect the ranking.

**Tokenizing numbers and codes.** Postgres's `simple` text search configuration splits `838,212` into `838` and `212`, and `28-1-5` into `28`, `-1` and `-5`. A generated `tsvector` column fixes this before splitting: thousands separators are removed, dashes between digits become dots, and digits glued to Persian letters (`مسافران28-1-1`) are separated. Words that contain digits get three times the weight, so an exact amount beats ordinary words in the same question.

**Why RRF with k = 1.** The usual default is k = 60, which flattens ranks (rank 1 scores 1/61, rank 10 scores 1/70). A number that only BM25 found at rank 1 then lost to chunks that both lists ranked in the middle. With k = 1, the top hit of each list stays near the top.

## Evaluation

Retrieval quality is measured with `app.cli.evaluate` on three hand-written question sets. Each question names the file and page that contain the answer. The metrics are hit@k (the right page is in the top k results) and MRR (mean reciprocal rank).

| Question set | Metric | Vector only | + BM25 with RRF | + Cohere rerank |
|---|---|---|---|---|
| Exact numbers and note codes | hit@1 | 0.429 | 0.533 | **1.000** |
| | hit@5 | 0.643 | 1.000 | 1.000 |
| | MRR | 0.501 | 0.756 | **1.000** |
| General questions (22) | hit@1 | 0.818 | 0.773 | **0.909** |
| | hit@5 | 1.000 | 1.000 | 1.000 |
| | MRR | 0.909 | 0.879 | 0.936 |
| Follow-ups after rewrite (20) | hit@1 | 0.950 | 0.850 | **1.000** |
| | hit@5 | 1.000 | 1.000 | 1.000 |

The exact-number set had 14 questions in the vector-only run and 15 afterwards. The added question is a long question with an exact number that the RRF-only setup missed.

The follow-up rewrite was measured on its own as well: hit@5 for the 20 follow-ups goes from 0.90 as typed to 1.00 after rewriting.

The question sets live in `data/eval/`, which is not committed because they refer to private test documents.

### Latency

Time from sending a question to the first word of the answer, measured inside the API container:

| | Before | After |
|---|---|---|
| New question | 5.8 s | 2.15 s |
| Follow-up question | 5.6 s | about 3.4 s |

These changes made the difference:

- **Minimal thinking** for the answer and the rewrite. Time to the first token went from 2.1 s to 1.1 s, and the rewrite from 1.9 s to 1.0 s, with the same facts and citations in the answers.
- **Reusing connections.** One Gemini client and one Cohere HTTP client per process, with idle connections kept for five minutes. A new TLS handshake on every request had made embedding take 1.3 to 1.5 s instead of 0.5 s, and the Cohere call 2.0 s instead of 0.7 s.

Vector and BM25 search together take about 20 ms for 172 chunks. Almost all remaining time is spent in calls to external APIs.

## Security

- Every search and every document query filters by the signed-in user. Requests for another user's document return 404, so document IDs of other users are not revealed.
- Passwords are hashed with scrypt. Unknown emails and wrong passwords get the same response and take the same time.
- Login is limited to 5 attempts per email per 15 minutes. Attempts are counted before the password is checked.
- Sessions are JWTs in an httpOnly, SameSite=Lax cookie, valid for 7 days.
- Uploaded file names are stored only as labels and never used as paths.
- The database is reachable only from other containers. All other ports are bound to `127.0.0.1`.
- Secrets are read from `.env`, which is ignored by git.

## Getting started

Requirements: Docker with Docker Compose, a [Gemini API key](https://aistudio.google.com/apikey), and optionally a [Cohere API key](https://dashboard.cohere.com/api-keys).

1. Create the environment file:

   ```bash
   cp .env.example .env
   ```

2. Fill in `GEMINI_API_KEY` and, if you want reranking, `COHERE_API_KEY`. Without a Cohere key the app uses the fused order.

3. Generate a random value for each of `POSTGRES_PASSWORD`, `JWT_SECRET` and `RABBITMQ_PASSWORD`:

   ```bash
   python3 -c "import secrets; print(secrets.token_hex(32))"
   ```

4. Start everything:

   ```bash
   docker compose up -d --build
   ```

5. Open http://localhost:3000, create an account and upload a file. Sample files are in `data/samples/`.

## Development

Run the backend tests (169 tests, against a real Postgres in a throwaway schema per test):

```bash
docker compose run --rm app pytest -q
```

Run the frontend tests:

```bash
docker compose exec frontend npm test
```

Run the retrieval evaluation:

```bash
docker compose run --rm app python -m app.cli.evaluate --questions data/eval/manual.jsonl --show-misses
```

Other command-line tools in `backend/app/cli/`:

| Module | Purpose |
|---|---|
| `inspect_file` | Show the extracted and normalized text of each page |
| `inspect_chunks` | Show how a file is chunked with a given size and overlap |
| `inspect_embeddings` | Compare similarity between texts, or search one file |
| `ingest` | Add, list or delete documents for a user without the web app |
| `search` | Run a vector search for a user |
| `ask` | Ask a question from the terminal, optionally printing the full prompt |
| `make_eval` | Generate a question set from stored chunks |

The API reloads on code changes. The worker does not, so restart it after changing task code:

```bash
docker compose restart worker
```

## Project structure

```
backend/app/
  api/        FastAPI routes: auth, documents, chat, conversations
  core/       Config, password hashing and tokens, user-facing messages
  db/         Postgres schema and all queries (store.py)
  services/   loaders, chunking, embeddings, ingestion, rag, rerank
  cli/        Command-line tools for inspection and evaluation
  worker.py   Celery app settings
  tasks.py    The embedding task
backend/tests/  pytest suite
frontend/app/   Next.js pages and components
```

## Known limitations

- **No OCR.** Scanned PDFs contain only images, so they are rejected with a message.
- **Not ready for public hosting.** The session cookie is not marked `Secure` because there is no HTTPS yet, and there is no per-IP rate limit.
- **Free API tiers.** Gemini's free tier limits embedding to about 100 texts per minute, so large files take a while. A Cohere trial key allows about 10 rerank calls per minute. Past that, answers fall back to the fused order.
- **BM25 scans the user's chunks on every query.** This is fast for thousands of chunks. Much larger collections would need a dedicated BM25 index.
- **Retrieval is measured, answer quality is not.** The evaluation checks whether the right page is found, not whether the generated answer is faithful to it.
