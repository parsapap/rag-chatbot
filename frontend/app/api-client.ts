export type Doc = {
  id: number;
  filename: string;
  pages: number;
  chunks: number;
  status: "processing" | "ready" | "failed";
  error: string | null;
};

const API_DOWN = "The API is not responding. Start it with: docker compose up -d api";

// Turns FastAPI's {"detail": ...} error body into a sentence for the user.
export function errorMessage(status: number, body: unknown): string {
  if (status === 409) return "You already uploaded this file.";
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  return status >= 500 ? API_DOWN : `Request failed (${status}).`;
}

export async function request(userId: string, path: string, init: RequestInit = {}): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, { ...init, headers: { ...init.headers, "X-User-Id": userId } });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new Error(API_DOWN);
  }
  if (!res.ok) throw new Error(errorMessage(res.status, await res.json().catch(() => null)));
  return res;
}

export async function api<T = unknown>(userId: string, path: string, init: RequestInit = {}): Promise<T> {
  const res = await request(userId, path, init);
  return (res.status === 204 ? null : await res.json()) as T;
}
