import type { User } from "./api-client";
import Login from "./login";

const STEPS = [
  { title: "Upload", body: "PDF, TXT or Markdown, up to 20 MB. Text is extracted, cleaned and split into chunks in the background." },
  { title: "Ask", body: "In Persian or English. Follow-up questions are rewritten using the conversation, so “when did it start?” works." },
  { title: "Check", body: "Every claim carries a numbered citation. Click it to read the exact passage, file and page." },
];

const FEATURES = [
  { label: "hybrid search", body: "Vector search for meaning plus BM25 for exact numbers and note codes, fused with reciprocal rank fusion." },
  { label: "reranking", body: "A cross-encoder reorders the top candidates, so the best passage comes first." },
  { label: "persian-ready", body: "Fixes reversed ligatures, merges table rows and normalizes digits before anything is indexed." },
  { label: "honest answers", body: "When your documents do not contain the answer, it says so instead of guessing." },
  { label: "private", body: "Each account searches only its own files. Nothing is shared between users." },
  { label: "measured", body: "Every retrieval change was checked on hand-written question sets before it shipped." },
];

const STATS = [
  { value: "0.95", label: "hit@1 on a held-out set of 56 questions" },
  { value: "2.2 s", label: "median time to the first word of an answer" },
  { value: "5", label: "passages retrieved and shown for every answer" },
];

export default function Landing({ onLogin }: { onLogin: (user: User) => void }) {
  return (
    <main className="col-span-full">
      <section className="relative overflow-hidden border-b border-border">
        <div aria-hidden="true" className="bg-grid pointer-events-none absolute inset-0" />
        <div className="relative mx-auto grid max-w-6xl items-center gap-12 px-6 py-20 lg:grid-cols-[1fr_auto] lg:py-28">
          <div className="flex flex-col gap-6">
            <p className="font-mono text-xs text-muted-foreground">// retrieval-augmented chat over your own files</p>
            <h1 className="max-w-2xl text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">
              Ask your documents. <span className="text-accent">See the page</span> every answer comes from.
            </h1>
            <p className="max-w-xl text-lg text-muted-foreground">
              Upload reports, statements or notes in Persian or English. Answers come only from your files, with a
              citation after every claim.
            </p>
            <p className="font-mono text-sm text-muted-foreground">
              <span className="text-foreground">Hybrid search.</span> <span className="text-foreground">Reranked.</span>{" "}
              <span className="text-foreground">Cited.</span>
            </p>
          </div>
          <Login onLogin={onLogin} />
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="mb-8 font-mono text-sm text-muted-foreground">// how it works</h2>
        <ol className="grid gap-px border border-border bg-border sm:grid-cols-3">
          {STEPS.map((s, i) => (
            <li key={s.title} className="flex flex-col gap-3 bg-background p-6">
              <span className="font-mono text-xs text-muted-foreground">{String(i + 1).padStart(2, "0")}</span>
              <h3 className="font-medium">{s.title}</h3>
              <p className="text-sm leading-relaxed text-muted-foreground">{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="border-y border-border bg-card">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="mb-8 font-mono text-sm text-muted-foreground">// under the hood</h2>
          <ul className="grid gap-px border border-border bg-border sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f) => (
              <li key={f.label} className="flex flex-col gap-2 bg-card p-6">
                <span className="font-mono text-xs text-accent">{f.label}</span>
                <p className="text-sm leading-relaxed text-muted-foreground">{f.body}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="mb-8 font-mono text-sm text-muted-foreground">// measured, not guessed</h2>
        <dl className="grid gap-8 sm:grid-cols-3">
          {STATS.map((s) => (
            <div key={s.label} className="flex flex-col gap-1 border-s-2 border-accent ps-4">
              <dt className="order-2 text-sm text-muted-foreground">{s.label}</dt>
              <dd className="order-1 font-mono text-3xl font-semibold">{s.value}</dd>
            </div>
          ))}
        </dl>
      </section>

      <footer className="border-t border-border">
        <p className="mx-auto max-w-6xl px-6 py-6 font-mono text-xs text-muted-foreground">
          Built with FastAPI, Postgres + pgvector, Gemini and Next.js.
        </p>
      </footer>
    </main>
  );
}
