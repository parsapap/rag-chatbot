"use client";

import { type FormEvent, useEffect, useRef, useState } from "react";
import { api, request } from "./api-client";
import { parseEvents, splitCitations } from "./chat-parsing";
import { plural } from "./documents";

type Source = { number: number; filename: string; page: number; score: number; content: string };
type Turn = {
  question: string;
  sources: Source[];
  text: string;
  done: { answered: boolean; cited: number[] } | null;
  error: string | null;
};
type SavedMessage = { question: string; answer: string; answered: boolean; sources: (Source & { cited: boolean })[] };

const toTurn = (m: SavedMessage): Turn => ({
  question: m.question,
  text: m.answer,
  sources: m.sources,
  done: { answered: m.answered, cited: m.sources.filter((s) => s.cited).map((s) => s.number) },
  error: null,
});

type Props = { conversationId: number | null; onSaved: (id: number) => void };

export default function Chat({ conversationId, onSaved }: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(conversationId !== null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<{ turn: number; source: number } | null>(null);
  const abort = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);
  // The server gives a new conversation its id with the first answer; follow-ups join it.
  const convId = useRef(conversationId);

  useEffect(() => () => abort.current?.abort(), []);

  useEffect(() => {
    if (conversationId === null) return;
    let live = true;
    api<{ messages: SavedMessage[] }>(`/conversations/${conversationId}`)
      .then((c) => live && setTurns(c.messages.map(toTurn)))
      .catch((e: Error) => live && setLoadError(e.message))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [conversationId]);
  useEffect(() => {
    if (turns.length) end.current?.scrollIntoView({ block: "end" });
  }, [turns.length]);

  const updateLast = (patch: (t: Turn) => Partial<Turn>) =>
    setTurns((ts) => [...ts.slice(0, -1), { ...ts[ts.length - 1], ...patch(ts[ts.length - 1]) }]);

  async function ask(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const question = String(new FormData(form).get("question")).trim();
    if (!question || streaming) return;
    form.reset();
    setTurns((ts) => [...ts, { question, sources: [], text: "", done: null, error: null }]);
    setSelected(null);
    setStreaming(true);
    const controller = new AbortController();
    abort.current = controller;
    let finished = false;
    try {
      const res = await request("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, conversation_id: convId.current }),
        signal: controller.signal,
      });
      const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const parsed = parseEvents(buffer + value);
        buffer = parsed.rest;
        for (const { event, data } of parsed.events) {
          if (event === "sources") updateLast(() => ({ sources: data as Source[] }));
          else if (event === "delta") updateLast((t) => ({ text: t.text + (data as { text: string }).text }));
          else if (event === "done") {
            finished = true;
            const d = data as { answered: boolean; cited: number[]; conversation_id: number };
            updateLast(() => ({ done: { answered: d.answered, cited: d.cited } }));
            convId.current = d.conversation_id;
            onSaved(d.conversation_id);
          } else if (event === "error") {
            finished = true;
            updateLast(() => ({ error: (data as { detail: string }).detail }));
          }
        }
      }
      if (!finished) updateLast(() => ({ error: "The connection closed before the answer finished. Ask again." }));
    } catch (err) {
      if ((err as Error).name !== "AbortError") updateLast(() => ({ error: (err as Error).message }));
    } finally {
      setStreaming(false);
    }
  }

  const shown = selected?.turn ?? turns.length - 1;
  const turn = turns[shown];
  const opened = selected?.turn === shown ? selected.source : turn?.done?.cited[0];

  return (
    <>
      <main className="flex min-w-0 flex-col border-border lg:border-x">
        <div className="flex flex-1 flex-col gap-10 p-8">
          {loading ? (
            <p className="text-muted-foreground">Loading conversation…</p>
          ) : loadError ? (
            <p role="alert" dir="auto" className="text-sm text-destructive">
              {loadError}
            </p>
          ) : (
            turns.length === 0 && (
              <p className="max-w-prose text-muted-foreground">
                Ask a question about your documents, in English or Persian. Every answer cites the page it comes
                from.
              </p>
            )
          )}
          {turns.map((t, i) => (
            <article key={i} aria-busy={streaming && i === turns.length - 1} className="flex flex-col gap-3">
              <p dir="auto" className="border-s-2 border-foreground ps-3.5 font-medium">
                {t.question}
              </p>
              {t.text ? (
                <p
                  dir="auto"
                  className={`max-w-[64ch] whitespace-pre-wrap text-[17px] leading-loose ${
                    t.done?.answered === false ? "text-muted-foreground" : ""
                  }`}
                >
                  {splitCitations(t.text).map((part, j) => {
                    if (typeof part === "string") return part;
                    const source = t.sources.find((s) => s.number === part);
                    if (!source) return `[${part}]`;
                    const active = i === shown && part === opened;
                    return (
                      <button
                        key={j}
                        type="button"
                        onClick={() => setSelected({ turn: i, source: part })}
                        aria-pressed={active}
                        aria-label={`Source ${part}: ${source.filename}, page ${source.page}`}
                        className={`mx-0.5 border px-1 align-baseline font-mono text-[13px] leading-normal ${
                          active ? "border-foreground bg-foreground text-background" : "border-input"
                        }`}
                      >
                        {part}
                      </button>
                    );
                  })}
                </p>
              ) : (
                !t.error && (
                  <p className="font-mono text-xs text-muted-foreground">
                    {t.sources.length ? "writing…" : "searching your documents…"}
                  </p>
                )
              )}
              {t.error && (
                <p role="alert" dir="auto" className="text-sm text-destructive">
                  {t.error}
                </p>
              )}
              {t.done && (
                <p className="font-mono text-xs text-muted-foreground">
                  {plural(t.sources.length, "source")} · {t.done.cited.length} cited
                </p>
              )}
            </article>
          ))}
          <div ref={end} />
        </div>

        <form onSubmit={ask} className="sticky bottom-0 flex gap-3 border-t border-border bg-background p-4">
          <input
            name="question"
            dir="auto"
            required
            maxLength={2000}
            disabled={streaming || loading || loadError !== null}
            autoComplete="off"
            aria-label="Question"
            placeholder="Ask about your documents…"
            className="min-w-0 flex-1 border border-input bg-card px-3 py-2 placeholder:text-muted-foreground"
          />
          <button
            type="submit"
            disabled={streaming || loading || loadError !== null}
            className="bg-foreground px-5 font-medium text-background disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
          >
            {streaming ? "Answering…" : "Ask"}
          </button>
        </form>
      </main>

      <aside className="flex flex-col gap-3 border-t border-border p-5 lg:sticky lg:top-0 lg:max-h-dvh lg:self-start lg:overflow-y-auto lg:border-t-0">
        <h2 className="border-b border-border pb-3 font-medium">Sources</h2>
        {!turn || (!turn.sources.length && !turn.done) ? (
          <p className="text-sm text-muted-foreground">The sources behind an answer show up here.</p>
        ) : (
          <>
            {turn.done && !turn.sources.length && (
              <p className="text-sm text-muted-foreground">You have no ready documents yet. Upload one on the left.</p>
            )}
            {turn.done?.answered === false && turn.sources.length > 0 && (
              <p className="text-sm text-muted-foreground">
                None of these passages answers the question. They are the closest matches found.
              </p>
            )}
            <ol className="flex flex-col gap-3">
              {turn.sources.map((s) => {
                const cited = turn.done?.cited.includes(s.number);
                const isOpen = s.number === opened;
                return (
                  <li key={s.number}>
                    <button
                      type="button"
                      onClick={() => setSelected({ turn: shown, source: s.number })}
                      aria-expanded={isOpen}
                      className={`w-full border bg-card p-3.5 text-start ${
                        isOpen ? "border-foreground" : "border-border"
                      } ${turn.done && !cited ? "text-muted-foreground" : ""}`}
                    >
                      <span className="flex justify-between gap-3 font-mono text-[13px]">
                        <span className="min-w-0 truncate">
                          [{s.number}] <bdi>{s.filename}</bdi> · p.{s.page}
                        </span>
                        <span className="shrink-0 text-xs text-muted-foreground">
                          {s.score.toFixed(3)}
                          {turn.done && !cited && " · not cited"}
                        </span>
                      </span>
                      <span className="mt-2.5 block h-0.5 bg-input">
                        <span
                          className="block h-full bg-foreground"
                          style={{ width: `${Math.max(0, Math.min(1, s.score)) * 100}%` }}
                        />
                      </span>
                      {isOpen && (
                        <span dir="auto" className="mt-3 block whitespace-pre-wrap text-[15px] leading-loose text-foreground">
                          {s.content}
                        </span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ol>
          </>
        )}
      </aside>
    </>
  );
}
