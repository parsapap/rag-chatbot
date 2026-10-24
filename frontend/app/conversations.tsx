"use client";

import { useEffect, useState } from "react";
import { api } from "./api-client";

type Conversation = { id: number; title: string; updated_at: string };

type Props = {
  activeId: number | null;
  version: number;
  onOpen: (id: number) => void;
  onNew: () => void;
};

export default function Conversations({ activeId, version, onOpen, onNew }: Props) {
  const [items, setItems] = useState<Conversation[] | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    let live = true;
    api<Conversation[]>("/conversations")
      .then((c) => {
        if (!live) return;
        setItems(c);
        setMessage(null);
      })
      .catch((e: Error) => live && setMessage(e.message));
    return () => {
      live = false;
    };
  }, [version, reloads]);

  async function remove(c: Conversation) {
    if (!confirm(`Delete the conversation "${c.title}"?`)) return;
    try {
      await api(`/conversations/${c.id}`, { method: "DELETE" });
      if (c.id === activeId) onNew();
      setReloads((v) => v + 1);
    } catch (e) {
      setMessage((e as Error).message);
    }
  }

  return (
    <section aria-labelledby="conversations-heading" className="flex flex-col gap-3 border-y border-border p-5 lg:border-b-0">
      <div className="flex items-baseline justify-between gap-3 border-b border-border pb-3">
        <h2 id="conversations-heading" className="font-medium">
          Conversations
        </h2>
        <button type="button" onClick={onNew} className="font-mono text-xs text-muted-foreground hover:text-foreground">
          + new
        </button>
      </div>

      {message && (
        <p role="alert" dir="auto" className="text-sm text-destructive">
          {message}
        </p>
      )}

      {items === null ? (
        !message && <p className="text-sm text-muted-foreground">Loading…</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-muted-foreground">No conversations yet. Your questions are saved here.</p>
      ) : (
        <ul>
          {items.map((c) => {
            const active = c.id === activeId;
            return (
              <li key={c.id} className="flex items-center gap-3 border-b border-border">
                <button
                  type="button"
                  onClick={() => onOpen(c.id)}
                  aria-current={active ? "true" : undefined}
                  className={`min-w-0 flex-1 py-2 text-start text-sm ${
                    active ? "font-medium text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <span dir="auto" className="block truncate">
                    {c.title}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => remove(c)}
                  aria-label={`Delete conversation ${c.title}`}
                  className="font-mono text-xs text-muted-foreground hover:text-destructive"
                >
                  delete
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
