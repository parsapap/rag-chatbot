"use client";

import { type FormEvent, useState } from "react";
import { api, type User } from "./api-client";

export default function Login({ onLogin }: { onLogin: (user: User) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const registering = mode === "register";

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    setBusy(true);
    setMessage(null);
    try {
      const user = await api<User>(`/auth/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: form.get("email"), password: form.get("password") }),
      });
      onLogin(user);
    } catch (err) {
      setMessage((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="col-span-full grid place-items-center p-8">
      <form onSubmit={submit} className="flex w-full max-w-sm flex-col gap-4 border border-border bg-card p-6">
        <h1 className="font-medium">{registering ? "Create an account" : "Log in"}</h1>
        <p className="text-sm text-muted-foreground">
          Your documents and conversations are private to your account.
        </p>

        <label className="flex flex-col gap-1.5 text-sm">
          Email
          <input
            name="email"
            type="email"
            required
            maxLength={254}
            autoComplete="email"
            className="border border-input bg-background px-3 py-2"
          />
        </label>
        <label className="flex flex-col gap-1.5 text-sm">
          Password
          <input
            name="password"
            type="password"
            required
            minLength={registering ? 8 : 1}
            maxLength={128}
            autoComplete={registering ? "new-password" : "current-password"}
            className="border border-input bg-background px-3 py-2"
          />
          {registering && <span className="text-xs text-muted-foreground">At least 8 characters.</span>}
        </label>

        {message && (
          <p role="alert" className="text-sm text-destructive">
            {message}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="bg-foreground py-2 font-medium text-background disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
        >
          {busy ? "Please wait…" : registering ? "Create account" : "Log in"}
        </button>
        <button
          type="button"
          onClick={() => {
            setMode(registering ? "login" : "register");
            setMessage(null);
          }}
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          {registering ? "Already have an account? Log in" : "No account yet? Create one"}
        </button>
      </form>
    </main>
  );
}
