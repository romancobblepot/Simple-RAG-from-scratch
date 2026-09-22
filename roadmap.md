# Roadmap — RAG Nutritional Chatbot

- [x] Store HF_TOKEN secret (GROQ_API_KEY already stored)
- [x] Server route `src/routes/api/chat.ts`: HF embeddings → Supabase `match_documents` RPC → Groq answer + sources
- [x] ARC Prize–inspired design system in `src/styles.css` (black, Space Mono, grid, ARC square colors)
- [x] Chat UI at `/`: sleek OpenAI-style, three-dot loader, thinking + done sounds
- [x] Clickable citations with popup showing highlighted source text
- [x] Header: "RAG Nutritional Chatbot: Build from Scratch" by Adnan Iqbal Kantroo (romancobblepot)
- [x] Head metadata + fonts in `__root.tsx`
- [x] Test API end-to-end, Playwright visual check
- [ ] Publish
