export type Doc = {
  id: number;
  filename: string;
  pages: number;
  chunks: number;
  status: "processing" | "ready" | "failed";
  error: string | null;
};

export type User = { id: number; email: string };

// Fired on any 401, so the workspace can switch to the login form when a session expires.
export const SESSION_ENDED = "session-ended";

const API_DOWN = "The API is not responding. Start it with: docker compose up -d api";

// Turns FastAPI's {"detail": ...} error body into a sentence for the user.
export function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (status === 409) return "You already uploaded this file.";
  return status >= 500 ? API_DOWN : `Request failed (${status}).`;
}

// The session cookie is httpOnly and same-origin, so the browser sends it on every call by itself.
export async function request(path: string, init: RequestInit = {}): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new Error(API_DOWN);
  }
  if (res.status === 401) window.dispatchEvent(new Event(SESSION_ENDED));
  if (!res.ok) throw new Error(errorMessage(res.status, await res.json().catch(() => null)));
  return res;
}

export async function api<T = unknown>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await request(path, init);
  return (res.status === 204 ? null : await res.json()) as T;
}
