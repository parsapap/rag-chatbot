import Workspace from "./workspace";

async function apiIsUp(): Promise<boolean> {
  try {
    const res = await fetch(`${process.env.API_URL}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(2000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export default async function Home() {
  const up = await apiIsUp();

  return (
    <div className="grid min-h-dvh grid-rows-[auto_1fr]">
      <header className="flex items-center justify-between border-b border-border px-5 py-3">
        <span className="font-medium">RAG Chatbot</span>
        {up ? (
          <span className="font-mono text-xs text-ok">● API connected</span>
        ) : (
          <span className="font-mono text-xs text-destructive">
            ✕ API unreachable. Start it with: docker compose up -d api
          </span>
        )}
      </header>

      <div className="grid min-h-0 lg:grid-cols-[300px_1fr_400px]">
        <Workspace />
      </div>
    </div>
  );
}
