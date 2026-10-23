"use client";

import { useState } from "react";
import Chat from "./chat";
import Documents from "./documents";

export default function Workspace() {
  const [userId, setUserId] = useState("demo_a");
  return (
    <>
      <Documents userId={userId} onUserChange={setUserId} />
      {/* key: switching user starts a fresh conversation and aborts a running answer. */}
      <Chat key={userId} userId={userId} />
    </>
  );
}
