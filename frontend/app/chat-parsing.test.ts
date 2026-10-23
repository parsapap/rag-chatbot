import assert from "node:assert/strict";
import test from "node:test";
import { parseEvents, splitCitations } from "./chat-parsing.ts";

test("parses complete events and keeps the unfinished tail", () => {
  const buffer =
    'event: sources\ndata: [{"number":1}]\n\n' +
    'event: delta\ndata: {"text":"فیدیبو"}\n\n' +
    "event: done\ndata: {";
  const { events, rest } = parseEvents(buffer);
  assert.deepEqual(events, [
    { event: "sources", data: [{ number: 1 }] },
    { event: "delta", data: { text: "فیدیبو" } },
  ]);
  assert.equal(rest, "event: done\ndata: {");
});

test("an event split across two chunks is parsed once both arrive", () => {
  const first = parseEvents('event: delta\ndata: {"te');
  assert.deepEqual(first.events, []);
  const second = parseEvents(first.rest + 'xt":"a"}\n\n');
  assert.deepEqual(second.events, [{ event: "delta", data: { text: "a" } }]);
  assert.equal(second.rest, "");
});

test("splits citations, including Persian and Arabic-Indic digits", () => {
  assert.deepEqual(splitCitations("الف [1]. ب [۲][٣] ج"), ["الف ", 1, ". ب ", 2, 3, " ج"]);
});

test("leaves text without citations alone", () => {
  assert.deepEqual(splitCitations("no sources [x] here"), ["no sources [x] here"]);
});
