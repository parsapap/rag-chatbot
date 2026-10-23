"use client";

import { useState } from "react";
import Chat from "./chat";
import Conversations from "./conversations";
import Documents from "./documents";

export default function Workspace() {
  const [userId, setUserId] = useState("demo_a");
  // n changes on every explicit open or "new", which remounts Chat; a new conversation
  // getting its id after the first answer only updates activeId, so the chat keeps its state.
  const [open, setOpen] = useState<{ id: number | null; n: number }>({ id: null, n: 0 });
  const [activeId, setActiveId] = useState<number | null>(null);
  const [listVersion, setListVersion] = useState(0);

  const show = (id: number | null) => {
    setOpen((o) => ({ id, n: o.n + 1 }));
    setActiveId(id);
  };

  return (
    <>
      <div className="flex min-w-0 flex-col">
        <Documents
          userId={userId}
          onUserChange={(id) => {
            setUserId(id);
            show(null);
          }}
        />
        <Conversations
          userId={userId}
          activeId={activeId}
          version={listVersion}
          onOpen={show}
          onNew={() => show(null)}
        />
      </div>
      {/* key: switching user or conversation aborts a running answer and starts clean. */}
      <Chat
        key={`${userId}:${open.n}`}
        userId={userId}
        conversationId={open.id}
        onSaved={(id) => {
          setActiveId(id);
          setListVersion((v) => v + 1);
        }}
      />
    </>
  );
}
