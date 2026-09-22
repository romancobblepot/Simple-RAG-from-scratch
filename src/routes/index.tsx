import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import {
  CitationDialog,
  type CitationGroup,
} from "@/components/chat/citation-dialog";
import { playAnswerSound, startThinkingSound } from "@/lib/sounds";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "RAG Nutritional Chatbot: Build from Scratch" },
      {
        name: "description",
        content:
          "A retrieval-augmented chatbot over the 2020 Human Nutrition textbook — 768-dim embeddings, hybrid vector + keyword search in Supabase, and an LLM. Built from scratch by Adnan Iqbal Kantroo.",
      },
      { property: "og:title", content: "RAG Nutritional Chatbot: Build from Scratch" },
      {
        property: "og:description",
        content:
          "Ask the Human Nutrition book anything — answers grounded in the 2020 edition with clickable page citations.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Index,
});

type Source = {
  page: number | null;
  source: string | null;
  similarity: number | null;
  content: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const STARTER_QUESTIONS = [
  "How often should infants be breastfed?",
  "What are the fat-soluble vitamins?",
  "What are the symptoms of pellagra?",
];

const LOGO_COLORS = [
  "bg-arc-blue",
  "bg-arc-red",
  "bg-arc-yellow",
  "bg-arc-green",
  "bg-foreground",
  "bg-arc-fuchsia",
  "bg-arc-orange",
  "bg-arc-teal",
  "bg-arc-grey",
];

function ArcLogo({ className = "h-7 w-7", gap = "gap-[2px]" }: { className?: string; gap?: string }) {
  return (
    <div className={`grid shrink-0 grid-cols-3 ${gap} ${className}`} aria-hidden>
      {LOGO_COLORS.map((color, i) => (
        <div key={i} className={`${color} h-full w-full`} />
      ))}
    </div>
  );
}

function groupSources(sources: Source[]): CitationGroup[] {
  const map = new Map<string, CitationGroup>();
  for (const s of sources) {
    const label = s.page != null ? `p. ${s.page}` : (s.source ?? "book");
    const existing = map.get(label);
    if (existing) {
      existing.chunks.push(s.content);
      if (s.similarity != null && (existing.similarity == null || s.similarity > existing.similarity)) {
        existing.similarity = s.similarity;
      }
    } else {
      map.set(label, {
        label,
        page: s.page,
        source: s.source ?? "Human Nutrition Book 2020",
        similarity: s.similarity,
        chunks: [s.content],
      });
    }
  }
  return [...map.values()];
}

function Index() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastQuery, setLastQuery] = useState("");
  const [activeCitation, setActiveCitation] = useState<CitationGroup | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, isLoading]);

  async function sendMessage(event?: FormEvent, preset?: string) {
    event?.preventDefault();
    const message = (preset ?? input).trim();
    if (!message || isLoading) return;

    setError("");
    setInput("");
    setLastQuery(message);
    setMessages((current) => [...current, { role: "user", content: message }]);
    setIsLoading(true);

    const stopThinkingSound = startThinkingSound();
    let answered = false;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Unable to get an answer.");

      setMessages((current) => [
        ...current,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
      answered = true;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      stopThinkingSound();
      setIsLoading(false);
      if (answered) playAnswerSound();
    }
  }

  function onInputKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  }

  return (
    <div className="relative flex min-h-screen flex-col bg-background font-mono text-foreground">
      {/* ARC-style hairline grid backdrop */}
      <div className="bg-grid-arc pointer-events-none fixed inset-0" aria-hidden />

      {/* Header */}
      <header className="relative z-10 border-b border-border bg-background/90 backdrop-blur-sm">
        <div className="mx-auto flex max-w-3xl items-center gap-3 px-4 py-3.5">
          <ArcLogo />
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-[13px] font-bold uppercase tracking-wider text-foreground">
              RAG Nutritional Chatbot{" "}
              <span className="text-muted-foreground">· Build from Scratch</span>
            </h1>
            <p className="mt-0.5 truncate text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              Made by Adnan Iqbal Kantroo{" "}
              <span className="text-primary">(romancobblepot)</span>
            </p>
          </div>
          <div className="hidden shrink-0 border border-border px-2 py-1 text-[9px] uppercase tracking-[0.2em] text-muted-foreground sm:block">
            Hybrid RAG · 768-dim
          </div>
        </div>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="relative z-10 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-8">
          {messages.length === 0 && (
            <div className="flex min-h-[55vh] flex-col items-center justify-center text-center">
              <div className="logo-pulse">
                <ArcLogo className="h-16 w-16" gap="gap-1" />
              </div>
              <h2 className="mt-8 text-xl font-bold uppercase tracking-wide text-foreground sm:text-2xl">
                Ask the Human Nutrition book
              </h2>
              <p className="mt-3 max-w-md text-xs leading-6 text-muted-foreground">
                Every answer is retrieved from the 2020 Human Nutrition textbook
                via embeddings + hybrid search. Educational information, not
                medical advice.
              </p>
              <div className="mt-8 flex max-w-lg flex-wrap justify-center gap-2">
                {STARTER_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    onClick={() => void sendMessage(undefined, question)}
                    disabled={isLoading}
                    className="border border-border bg-card px-4 py-2.5 text-left text-xs text-card-foreground transition-colors hover:border-primary hover:text-primary disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-8">
            {messages.map((message, index) => (
              <article key={index}>
                {message.role === "user" ? (
                  <div className="flex justify-end">
                    <div className="max-w-[85%] border border-border bg-card px-4 py-3">
                      <p className="text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
                        You
                      </p>
                      <p className="mt-1 whitespace-pre-wrap text-sm leading-6 text-card-foreground">
                        {message.content}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-2 bg-primary" aria-hidden />
                      <p className="text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
                        Nutrition Bot
                      </p>
                    </div>
                    <p className="mt-2 whitespace-pre-wrap text-sm leading-7 text-foreground">
                      {message.content}
                    </p>

                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-4">
                        <p className="text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
                          Citations — click to inspect
                        </p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {groupSources(message.sources).map((citation) => (
                            <button
                              key={citation.label}
                              onClick={() => setActiveCitation(citation)}
                              className="group flex items-center gap-2 border border-border bg-card px-2.5 py-1.5 text-[11px] text-muted-foreground transition-colors hover:border-primary hover:text-primary"
                            >
                              <span className="h-1.5 w-1.5 bg-primary" aria-hidden />
                              <span className="font-bold">[ {citation.label} ]</span>
                              {citation.similarity != null && (
                                <span className="text-[9px] uppercase tracking-widest opacity-60 group-hover:opacity-100">
                                  {Math.round(citation.similarity * 100)}%
                                </span>
                              )}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </article>
            ))}

            {isLoading && (
              <div>
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 animate-pulse bg-primary" aria-hidden />
                  <p className="text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
                    Retrieving + generating
                  </p>
                </div>
                <div className="mt-3 flex items-center gap-2" aria-label="Thinking">
                  <span className="thinking-dot" />
                  <span className="thinking-dot" style={{ animationDelay: "0.15s" }} />
                  <span className="thinking-dot" style={{ animationDelay: "0.3s" }} />
                </div>
              </div>
            )}
          </div>

          <div className="h-6" />
        </div>
      </div>

      {/* Input */}
      <div className="relative z-10 border-t border-border bg-background/90 backdrop-blur-sm">
        <form
          onSubmit={(e) => void sendMessage(e)}
          className="mx-auto max-w-3xl px-4 py-4"
        >
          {error && (
            <p className="mb-3 border border-destructive/50 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {error}
            </p>
          )}
          <div className="flex items-end gap-2 border border-border bg-card px-3 py-2 transition-colors focus-within:border-primary">
            <label htmlFor="question" className="sr-only">
              Ask a nutrition question
            </label>
            <textarea
              id="question"
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onInputKeyDown}
              placeholder="Ask a question about nutrition…"
              disabled={isLoading}
              className="max-h-40 min-w-0 flex-1 resize-none bg-transparent px-1 py-1.5 text-sm text-foreground outline-none placeholder:text-muted-foreground disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="shrink-0 bg-primary px-4 py-2 text-xs font-bold uppercase tracking-widest text-primary-foreground transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-30"
            >
              Send
            </button>
          </div>
          <p className="mt-2 text-center text-[9px] uppercase tracking-[0.25em] text-muted-foreground">
            Grounded in Human Nutrition · 2020 Edition — not medical advice
          </p>
        </form>
      </div>

      <CitationDialog
        citation={activeCitation}
        query={lastQuery}
        onClose={() => setActiveCitation(null)}
      />
    </div>
  );
}
