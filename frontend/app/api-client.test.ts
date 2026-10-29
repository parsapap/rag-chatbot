import assert from "node:assert/strict";
import test from "node:test";
import { type Doc, errorMessage, statusLabel } from "./api-client.ts";

test("uses the API's string detail", () => {
  assert.equal(errorMessage(413, { detail: "file larger than 20 MB" }), "file larger than 20 MB");
});

test("explains a duplicate instead of showing the raw object", () => {
  const body = { detail: { message: "duplicate file", document_id: 3 } };
  assert.equal(errorMessage(409, body), "You already uploaded this file.");
});

test("an unreadable 5xx body means the API is down", () => {
  assert.match(errorMessage(500, null), /not responding/);
});

test("other unreadable errors show the status", () => {
  assert.equal(errorMessage(404, null), "Request failed (404).");
});

const doc = (status: Doc["status"], total: number, embedded: number | null): Doc => ({
  id: 1, filename: "a.pdf", pages: 1, chunks: 0, status, error: null, total_chunks: total, embedded_chunks: embedded,
});

test("a document the worker has not started is queued", () => {
  assert.equal(statusLabel(doc("processing", 116, null)), "queued");
});

test("progress is a whole percentage that stays below 100 until ready", () => {
  assert.equal(statusLabel(doc("processing", 116, 0)), "embedding 0%");
  assert.equal(statusLabel(doc("processing", 116, 50)), "embedding 43%");
  assert.equal(statusLabel(doc("processing", 116, 116)), "embedding 99%");
  assert.equal(statusLabel(doc("ready", 116, 116)), "ready");
});
