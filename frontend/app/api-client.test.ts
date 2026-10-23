import assert from "node:assert/strict";
import test from "node:test";
import { errorMessage } from "./api-client.ts";

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
