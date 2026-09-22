import { useEffect, type ReactNode } from "react";

export type CitationGroup = {
  label: string;
  page: number | null;
  source: string;
  similarity: number | null;
  chunks: string[];
};

const STOP_WORDS = new Set([
  "the", "and", "are", "what", "how", "for", "with", "should", "you", "your",
  "does", "this", "that", "from", "have", "has", "why", "when", "which",
  "can", "all", "any", "much", "many", "often",
]);

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Wraps query terms found in the excerpt in highlighted <mark> elements. */
function highlight(text: string, query: string): ReactNode {
  const terms = query
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 2 && !STOP_WORDS.has(t));
  if (!terms.length) return text;

  const re = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "gi");
  const parts = text.split(re);
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="bg-primary/25 text-primary">
        {part}
      </mark>
    ) : (
      part
    ),
  );
}

export function CitationDialog({
  citation,
  query,
  onClose,
}: {
  citation: CitationGroup | null;
  query: string;
  onClose: () => void;
}) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  if (!citation) return null;

  const similarityPct =
    citation.similarity != null ? Math.round(citation.similarity * 100) : null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label={`Citation ${citation.label}`}
    >
      <button
        aria-label="Close citation"
        onClick={onClose}
        className="absolute inset-0 cursor-pointer bg-background/85 backdrop-blur-sm"
      />
      <div className="relative flex max-h-[80vh] w-full max-w-xl flex-col border border-border bg-card shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
              Source excerpt
            </p>
            <p className="mt-1 truncate text-sm font-bold text-foreground">
              {citation.source}
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span className="border border-primary/50 px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest text-primary">
                {citation.label}
              </span>
              {similarityPct != null && (
                <span className="border border-border px-2 py-0.5 text-[10px] uppercase tracking-widest text-muted-foreground">
                  {similarityPct}% match
                </span>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="shrink-0 border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:border-foreground hover:text-foreground"
          >
            ✕
          </button>
        </div>

        {/* Excerpts */}
        <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
          {citation.chunks.map((chunk, i) => (
            <div key={i}>
              <p className="mb-1.5 text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
                Excerpt {citation.chunks.length > 1 ? i + 1 : ""} · retrieved
                from the book
              </p>
              <blockquote className="border-l-2 border-primary/50 pl-3 text-xs leading-6 text-card-foreground">
                {highlight(chunk, query)}
              </blockquote>
            </div>
          ))}
        </div>

        <div className="border-t border-border px-5 py-3">
          <p className="text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
            Human Nutrition · 2020 Edition — educational use only
          </p>
        </div>
      </div>
    </div>
  );
}
