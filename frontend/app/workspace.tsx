"use client";

import { useEffect, useState } from "react";
import { api, SESSION_ENDED, type User } from "./api-client";
import Chat from "./chat";
import Conversations from "./conversations";
import Documents from "./documents";
import Landing from "./landing";

// The open conversation lives in the URL (/c/12), so a refresh, a bookmark or Back reopens it.
const pathFor = (id: number | null) => (id === null ? "/" : `/c/${id}`);
const idFromPath = () => {
  const m = window.location.pathname.match(/^\/c\/(\d+)$/);
  return m ? Number(m[1]) : null;
};
const setPath = (id: number | null, mode: "push" | "replace") => {
  if (window.location.pathname === pathFor(id)) return;
  if (mode === "push") window.history.pushState(null, "", pathFor(id));
  else window.history.replaceState(null, "", pathFor(id));
};

export default function Workspace() {
  // undefined while the session is being checked, null when logged out.
  const [user, setUser] = useState<User | null | undefined>(undefined);
  // n changes on every explicit open or "new", which remounts Chat; a new conversation
  // getting its id after the first answer only updates activeId, so the chat keeps its state.
  const [open, setOpen] = useState<{ id: number | null; n: number }>({ id: null, n: 0 });
  const [activeId, setActiveId] = useState<number | null>(null);
  const [listVersion, setListVersion] = useState(0);

  const show = (id: number | null) => {
    setOpen((o) => ({ id, n: o.n + 1 }));
    setActiveId(id);
  };

  const navigate = (id: number | null) => {
    setPath(id, "push");
    show(id);
  };

  useEffect(() => {
    show(idFromPath());
    api<User>("/auth/me")
      .then(setUser)
      .catch(() => setUser(null));
    const ended = () => setUser(null);
    const moved = () => show(idFromPath());
    window.addEventListener(SESSION_ENDED, ended);
    window.addEventListener("popstate", moved);
    return () => {
      window.removeEventListener(SESSION_ENDED, ended);
      window.removeEventListener("popstate", moved);
    };
  }, []);

  // Keeps the URL: after an expired session, signing in again returns to the same conversation.
  const login = (u: User) => {
    show(idFromPath());
    setUser(u);
  };

  async function logout() {
    await api("/auth/logout", { method: "POST" }).catch(() => null);
    setPath(null, "replace");
    setUser(null);
  }

  if (user === undefined) return <p className="col-span-full p-8 text-muted-foreground">Loading…</p>;
  if (user === null) return <Landing onLogin={login} />;

  return (
    <>
      <div className="flex min-w-0 flex-col">
        <Documents key={user.id} email={user.email} onLogout={logout} />
        <Conversations key={`c${user.id}`} activeId={activeId} version={listVersion} onOpen={navigate} onNew={() => navigate(null)} />
      </div>
      {/* key: switching account or conversation aborts a running answer and starts clean. */}
      <Chat
        key={`${user.id}:${open.n}`}
        conversationId={open.id}
        onSaved={(id) => {
          // replace, not push: Back should not return to the empty chat this conversation started as.
          setPath(id, "replace");
          setActiveId(id);
          setListVersion((v) => v + 1);
        }}
      />
    </>
  );
}
