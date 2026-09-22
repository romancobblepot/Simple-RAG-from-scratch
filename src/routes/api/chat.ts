import { createFileRoute } from "@tanstack/react-router";

type RetrievedChunk = {
  doc_id: string;
  chunk_index: number;
  content: string;
  metadata: { page?: number; source?: string } | null;
  similarity: number;
};

const SYSTEM_PROMPT =
  "You are a helpful nutrition-book assistant. Answer only from the provided context. " +
  "If the context does not contain the answer, say so plainly. Do not reveal reasoning or invent facts. " +
  "This is educational information, not medical diagnosis. Keep answers concise.";

/** Mean-pool nested embedding output down to a flat 768-dim vector. */
function toFlatEmbedding(raw: unknown): number[] {
  let arr = raw as unknown[];
  while (Array.isArray(arr) && Array.isArray(arr[0])) {
    const matrix = arr as number[][];
    const width = matrix[0]?.length ?? 0;
    const pooled = new Array<number>(width).fill(0);
    for (const row of matrix) {
      for (let i = 0; i < width; i++) pooled[i] = (pooled[i] ?? 0) + (row[i] ?? 0);
    }
    arr = pooled.map((v) => v / (matrix.length || 1));
  }
  return arr as number[];
}

/** Qwen-family models can emit <think> blocks; never show them. */
function stripReasoning(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<think>[\s\S]*$/g, "")
    .trim();
}

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        try {
          const GROQ_API_KEY = process.env["GROQ_API_KEY"];
          const HF_TOKEN = process.env["HF_TOKEN"];
          if (!GROQ_API_KEY || !HF_TOKEN) {
            return Response.json(
              { error: "Server is missing API configuration." },
              { status: 500 },
            );
          }

          const body = (await request.json().catch(() => null)) as
            | { message?: unknown }
            | null;
          const message =
            typeof body?.message === "string" ? body.message.trim() : "";

          if (!message) {
            return Response.json({ error: "message is required" }, { status: 400 });
          }
          if (message.length > 2000) {
            return Response.json({ error: "message is too long" }, { status: 400 });
          }

          // 1. Embed the query (768-dim, matches the pgvector column).
          const { InferenceClient } = await import("@huggingface/inference");
          const hf = new InferenceClient(HF_TOKEN);
          const rawEmbedding = await hf.featureExtraction({
            model: "sentence-transformers/all-mpnet-base-v2",
            inputs: message,
          });
          const embedding = toFlatEmbedding(rawEmbedding);

          // 2. Hybrid retrieval from Supabase (0.7 vector + 0.3 keyword).
          const { supabaseAdmin } = await import(
            "@/integrations/supabase/client.server"
          );
          const { data, error } = await supabaseAdmin.rpc("match_documents", {
            query_embedding: `[${embedding.join(",")}]`,
            query_text: message,
            match_count: 4,
          });
          if (error) throw error;

          const chunks = (data ?? []) as RetrievedChunk[];
          if (!chunks.length) {
            return Response.json({
              answer:
                "I couldn't find relevant information in the nutrition book. Try rephrasing your question.",
              sources: [],
            });
          }

          const context = chunks
            .map((chunk) => {
              const source = chunk.metadata?.source ?? "Human Nutrition Book";
              const page = chunk.metadata?.page ?? "?";
              return `[${source}, page ${page}]\n${chunk.content.slice(0, 3500)}`;
            })
            .join("\n\n");

          // 3. Generate the grounded answer with Groq.
          const groqRes = await fetch(
            "https://api.groq.com/openai/v1/chat/completions",
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${GROQ_API_KEY}`,
              },
              body: JSON.stringify({
                model: "qwen/qwen3.8-27b",
                temperature: 0.2,
                max_tokens: 600,
                messages: [
                  { role: "system", content: SYSTEM_PROMPT },
                  {
                    role: "user",
                    content: `Context:\n${context}\n\nQuestion: ${message}`,
                  },
                ],
              }),
            },
          );

          if (!groqRes.ok) {
            const detail = await groqRes.text();
            console.error("Groq error", groqRes.status, detail);
            throw new Error(`LLM request failed (${groqRes.status})`);
          }

          const completion = (await groqRes.json()) as {
            choices?: { message?: { content?: string } }[];
          };
          const answer =
            stripReasoning(completion.choices?.[0]?.message?.content ?? "") ||
            "I couldn't generate an answer.";

          return Response.json({
            answer,
            sources: chunks.map((chunk) => ({
              page: chunk.metadata?.page ?? null,
              source: chunk.metadata?.source ?? "Human Nutrition Book 2020",
              similarity: chunk.similarity ?? null,
              content: chunk.content,
            })),
          });
        } catch (error) {
          console.error("Chat route error:", error);
          return Response.json(
            { error: "Unable to answer the question right now." },
            { status: 500 },
          );
        }
      },
    },
  },
});
