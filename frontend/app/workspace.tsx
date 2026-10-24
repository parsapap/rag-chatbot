"use client";

import { useEffect, useState } from "react";
import { api, SESSION_ENDED, type User } from "./api-client";
import Chat from "./chat";
import Conversations from "./conversations";
import Documents from "./documents";
import Login from "./login";

export default function Workspace() {
  // undefined while the session is being checked, null when logged out.
  const [user, setUser] = useState<User | null | undefined>(undefined);
  // n changes on every explicit open or "new", which remounts Chat; a new conversation
  // getting its id after the first answer only updates activeId, so the chat keeps its state.
  const [open, setOpen] = useState<{ id: number | null; n: number }>({ id: null, n: 0 });
  const [activeId, setActiveId] = useState<number | null>(null);
  const [listVersion, setListVersion] = useState(0);

  useEffect(() => {
    api<User>("/auth/me")
      .then(setUser)
      .catch(() => setUser(null));
    const ended = () => setUser(null);
    window.addEventListener(SESSION_ENDED, ended);
    return () => window.removeEventListener(SESSION_ENDED, ended);
  }, []);

  const show = (id: number | null) => {
    setOpen((o) => ({ id, n: o.n + 1 }));
    setActiveId(id);
  };

  const login = (u: User) => {
    show(null);
    setUser(u);
  };

  async function logout() {
    await api("/auth/logout", { method: "POST" }).catch(() => null);
    setUser(null);
  }

  if (user === undefined) return <p className="col-span-full p-8 text-muted-foreground">Loading…</p>;
  if (user === null) return <Login onLogin={login} />;

  return (
    <>
      <div className="flex min-w-0 flex-col">
        <Documents key={user.id} email={user.email} onLogout={logout} />
        <Conversations key={`c${user.id}`} activeId={activeId} version={listVersion} onOpen={show} onNew={() => show(null)} />
      </div>
      {/* key: switching account or conversation aborts a running answer and starts clean. */}
      <Chat
        key={`${user.id}:${open.n}`}
        conversationId={open.id}
        onSaved={(id) => {
          setActiveId(id);
          setListVersion((v) => v + 1);
        }}
      />
    </>
  );
}
