# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Next.js (App Router) + Tailwind CSS + shadcn/ui, chosen by the user in the original project plan. Runs as a `frontend` service in the existing `docker-compose.yml`, bound to 127.0.0.1 like the API. Backend is the FastAPI app in `backend/`.

## Users

The builder, a developer learning RAG, using it themself and showing it as a portfolio piece (GitHub, CV). Reads and writes Persian. The interface is in English, left to right; documents, questions and answers may be Persian, and that content renders right to left.

## Product Purpose

Ask questions, in Persian or English, about your own uploaded documents (PDF, TXT, MD) and get answers drawn only from them, with the page each claim came from. Success: an answer you can check against the source in one click, and an honest "not in your documents" when the answer is not there.

## Positioning

Every sentence of an answer points to a numbered source (file and page) retrieved from the user's own files; the model is instructed not to answer from its own knowledge. Each user only ever searches their own documents.

## Operating Context

- Documents on hand are Persian business reports and an interim financial statement: prose, KPI pages, and dense number tables (amounts like 539,008 million rials, Jalali dates like 1404/09/30).
- Upload returns at once; embedding runs in the background and takes seconds to about two minutes on the free Gemini quota. A document moves through `processing`, `ready`, `failed` (with a reason).
- Answers stream in pieces over SSE: sources first, then text, then which sources were cited.

## Capabilities and Constraints

- API: `POST/GET/DELETE /documents`, `GET /documents/{id}`, `POST /chat`, `POST /chat/stream` (events `sources`, `delta`, `done`, `error`). See `backend/app/api/`.
- Upload limits: .pdf/.txt/.md, 20 MB, duplicates per user rejected (409), scanned PDFs rejected (no OCR).
- Retrieval returns 1 to 10 sources (default 5); citations are `[n]` in the answer text, matching source `number`.
- Auth is a placeholder: the client sends `X-User-Id`. Real login/register comes in a later phase; until then the UI only lets the user pick a user id, and nothing is deployed publicly.
- Model output is not always consistent: Persian vs ASCII digits, and multi-column PDF pages can mix list order.
- Light and dark themes, following the system setting.
- Undecided: chat history (not stored yet), multiple conversations.

## Evidence on Hand

Real sample documents in `data/fa_docs/` (gitignored, personal) and `data/samples/`. No users, testimonials, or metrics exist; do not invent them.

## Product Principles

1. The source is the proof: every answer shows where it came from, and citations are easy to open.
2. Saying "not found" is a correct answer, not a failure state.
3. Show background work honestly: processing, failure reasons, and quota limits are stated plainly.
4. English interface, Persian-ready content: interface text is English; document text, questions and answers keep their own direction (`dir="auto"`) and read naturally in Persian.

## Accessibility & Inclusion

Left-to-right interface with per-element direction for mixed Persian/English content, keyboard reachable, readable contrast in both themes, reduced motion respected.
