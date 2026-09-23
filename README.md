# RAG Nutritional Chatbot — Build from Scratch

A full-stack **Retrieval-Augmented Generation (RAG)** chatbot that answers
nutrition questions strictly from the *Human Nutrition — 2020 Edition*
textbook. Every answer is grounded in retrieved book chunks and ships with
**clickable citations** that open the exact passage the answer came from.

Built by **Adnan Iqbal Kantroo** ([@romancobblepot](https://github.com/romancobblepot)).

**Live:** https://wisdom-well-nutri.lovable.app

---

## Architecture

```text
                 ┌────────────────────────── INGESTION (offline) ──────────────────────────┐
                 │                                                                          │
  PDF textbook ─▶ ingest.py ─▶ sentence chunks (10 sentences, ≥30 tokens)                   │
                                 │                                                          │
                                 ▼                                                          │
                    all-mpnet-base-v2 (768-dim embeddings, local)                           │
                                 │                                                          │
                                 ▼                                                          │
                    Supabase Postgres + pgvector  (public.chunks)                           │
                 └──────────────────────────────────────────────────────────────────────────┘

                 ┌────────────────────────── QUERY TIME (per message) ──────────────────────┐
                 │                                                                          │
  User question ─▶ POST /api/chat                                                          │
                      │                                                                    │
                      ├─ 1. Embed query  → Hugging Face Inference API (all-mpnet-base-v2)  │
                      ├─ 2. Retrieve     → Supabase RPC match_documents()                  │
                      │      hybrid score = 0.7 × cosine similarity + 0.3 × ts_rank_cd     │
                      ├─ 3. Generate     → Groq (OpenAI-compatible), qwen/qwen3.8-27b      │
                      │      system prompt constrains answers to the retrieved context     │
                      └─ 4. Respond      → { answer, sources[] }                            │
                 │                                                                          │
  React chat UI ◀── answer + clickable citation chips (page-level)                         │
                 └──────────────────────────────────────────────────────────────────────────┘
```

## The RAG pipeline

### 1. Ingestion — `ingest.py`

Offline Python script that builds the knowledge base:

1. **Extract** — reads the PDF with PyMuPDF, normalizes whitespace/hyphenation.
2. **Chunk** — spaCy `sentencizer` splits pages into sentences, grouped into
   **chunks of 10 sentences**; chunks under **30 tokens** are dropped.
3. **Embed** — each chunk is encoded locally with
   `sentence-transformers/all-mpnet-base-v2` → **768-dimensional vectors**.
4. **Upload** — rows are batch-inserted (200/batch) into `public.chunks` with
   metadata `{ page, source }` and `doc_id = "nutrition-v1"`.

### 2. Storage & hybrid retrieval — Supabase / pgvector

```sql
public.chunks (doc_id, chunk_index, content, metadata jsonb, embedding vector(768))
```

- **IVFFlat index** on `embedding` (cosine ops) for fast ANN search.
- **GIN index** on `to_tsvector('english', content)` for keyword search.
- **`match_documents(query_embedding, query_text, match_count, filter)`** RPC
  scores every chunk with a hybrid blend:

```text
score = 0.7 × (1 − cosine_distance)  +  0.3 × ts_rank_cd(FTS match)
```

So answers benefit from both **semantic meaning** and **exact keyword overlap**.

### 3. Answer generation — Groq LLM

Retrieved chunks are formatted as `[source, page N]` context blocks
(truncated to 3 500 chars each) and sent to Groq:

| Setting | Value |
|---|---|
| Model | `qwen/qwen3.8-27b` |
| Temperature | `0.2` (low — factual, grounded) |
| Max tokens | `600` |
| System prompt | Answer **only** from the provided context; say so plainly when the context lacks the answer; never invent facts; educational info, not medical advice |

Any `<think>` reasoning blocks the model emits are stripped before display.

## Chatbot features

- **Minimal, sleek chat UI** styled after the [ARC Prize](https://arcprize.org/arc-agi)
  aesthetic — pure black background, Space Mono type, hairline grid, colored-square logo.
- **Three-dot thinking indicator** while the pipeline retrieves and generates.
- **Subtle sounds** — a soft tick loop while thinking, a two-note chime when the
  answer lands (WebAudio, no audio files).
- **Clickable citations** — each answer lists the book pages used; clicking one
  opens a popup showing the exact retrieved passage with the query terms highlighted.
- **Starter questions** to try the pipeline instantly.
- Server-side only keys — the service-role key and LLM keys never reach the browser.

## Tech stack

| Layer | Choice |
|---|---|
| Frontend | React 19 + TanStack Start, Tailwind CSS v4, Space Mono |
| API route | TanStack Start server route (`src/routes/api/chat.ts`) |
| Embeddings (ingest) | `sentence-transformers/all-mpnet-base-v2` (local, PyTorch) |
| Embeddings (query) | Hugging Face Inference API, same model — vectors stay compatible |
| Vector DB | Supabase Postgres + pgvector (`vector(768)`, IVFFlat + GIN FTS) |
| LLM | Groq OpenAI-compatible API — `qwen/qwen3.8-27b` |
| Citations | Page-level metadata returned with every answer |

## Environment variables

```bash
SUPABASE_URL=...                    # project URL
SUPABASE_SERVICE_ROLE_KEY=...       # server-only (ingest + API route)
HF_TOKEN=...                        # Hugging Face Inference API (query embeddings)
GROQ_API_KEY=...                    # Groq LLM
```

## Running locally

```bash
# 1. Install & start the web app
bun install
bun run dev          # http://localhost:8080

# 2. (Re)build the knowledge base — only needed for a new/changed PDF
pip install pymupdf spacy sentence-transformers supabase pandas tqdm python-dotenv
python ingest.py     # edit doc_path first
```

## Project layout

```text
ingest.py                              # offline ingestion pipeline (PDF → chunks → embeddings → Supabase)
src/routes/api/chat.ts                 # RAG endpoint: embed → retrieve (match_documents) → Groq → answer + sources
src/routes/index.tsx                   # chat interface (ARC-styled)
src/components/chat/citation-dialog.tsx# clickable citation popup with highlighted excerpt
src/lib/sounds.ts                      # WebAudio thinking/answer sounds
src/styles.css                         # ARC theme tokens (black, Space Mono, hairline grid)
```

---

*Educational information only — not medical advice.*
