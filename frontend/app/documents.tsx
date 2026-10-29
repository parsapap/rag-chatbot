"use client";

import { useEffect, useState } from "react";
import { api, type Doc, statusLabel } from "./api-client";

const POLL_MS = 2000;

export const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

const STATUS = {
  ready: { glyph: "●", className: "text-ok" },
  processing: { glyph: "◌", className: "text-wait" },
  failed: { glyph: "✕", className: "text-destructive" },
} as const;

export default function Documents({ email, onLogout }: { email: string; onLogout: () => void }) {
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [version, setVersion] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const reload = () => setVersion((v) => v + 1);

  useEffect(() => {
    let live = true;
    api<Doc[]>("/documents")
      .then((d) => live && setDocs(d))
      .catch((e: Error) => live && setMessage(e.message));
    return () => {
      live = false;
    };
  }, [version]);

  // Embedding runs in the background on the API, so poll until nothing is processing.
  useEffect(() => {
    if (!docs?.some((d) => d.status === "processing")) return;
    const t = setTimeout(reload, POLL_MS);
    return () => clearTimeout(t);
  }, [docs]);

  async function upload(file: File | undefined) {
    if (!file || busy) return;
    setBusy(true);
    setMessage(null);
    const form = new FormData();
    form.append("file", file);
    try {
      await api("/documents", { method: "POST", body: form });
      reload();
    } catch (e) {
      setMessage((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(doc: Doc) {
    if (!confirm(`Delete ${doc.filename}? Its chunks are removed too.`)) return;
    setMessage(null);
    try {
      await api(`/documents/${doc.id}`, { method: "DELETE" });
      reload();
    } catch (e) {
      setMessage((e as Error).message);
    }
  }

  return (
    <aside className="flex flex-col gap-4 border-b border-border p-5 lg:border-b-0">
      <div className="flex items-baseline justify-between gap-3 border-b border-border pb-3">
        <h2 className="font-medium">Documents</h2>
        <span className="flex min-w-0 items-baseline gap-2 font-mono text-xs text-muted-foreground">
          <span title={email} className="min-w-0 truncate">
            {email}
          </span>
          <button type="button" onClick={onLogout} className="shrink-0 hover:text-foreground">
            log out
          </button>
        </span>
      </div>

      <label
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          upload(e.dataTransfer.files[0]);
        }}
        className="block cursor-pointer border border-dashed border-input p-5 text-center text-sm text-muted-foreground focus-within:outline-2 focus-within:outline-foreground"
      >
        <span className="block font-medium text-foreground">{busy ? "Uploading…" : "Add a document"}</span>
        Drop a file here or click to choose. PDF, TXT or MD, up to 20 MB.
        <input
          type="file"
          accept=".pdf,.txt,.md"
          disabled={busy}
          className="sr-only"
          onChange={(e) => {
            upload(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </label>

      {message && (
        <p role="alert" dir="auto" className="text-sm text-destructive">
          {message}
        </p>
      )}

      {docs === null ? (
        !message && <p className="text-sm text-muted-foreground">Loading…</p>
      ) : docs.length === 0 ? (
        <p className="text-sm text-muted-foreground">No documents yet.</p>
      ) : (
        <ul>
          {docs.map((d) => (
            <li key={d.id} className="grid grid-cols-[1fr_auto] gap-x-3 border-b border-border py-2.5">
              <span dir="auto" title={d.filename} className="min-w-0 truncate font-mono text-[13px]">
                {d.filename}
              </span>
              <span className={`font-mono text-xs ${STATUS[d.status].className}`}>
                <span aria-hidden="true">{STATUS[d.status].glyph}</span> {statusLabel(d)}
              </span>
              <span className="font-mono text-xs text-muted-foreground">
                {plural(d.pages, "page")}
                {d.status === "ready" && ` · ${plural(d.chunks, "chunk")}`}
              </span>
              <button
                type="button"
                onClick={() => remove(d)}
                aria-label={`Delete ${d.filename}`}
                className="justify-self-end font-mono text-xs text-muted-foreground hover:text-destructive"
              >
                delete
              </button>
              {d.error && (
                <span dir="auto" className="col-span-2 pt-1 text-xs text-destructive">
                  {d.error}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
