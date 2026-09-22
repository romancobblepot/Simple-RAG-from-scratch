# Roadmap — RAG Nutritional Chatbot

- [x] Store HF_TOKEN secret (GROQ_API_KEY already stored)
- [ ] Server route `src/routes/api/chat.ts`: HF embeddings → Supabase `match_documents` RPC → Groq answer + sources
- [ ] ARC Prize–inspired design system in `src/styles.css` (black, Space Mono, grid, ARC square colors)
- [ ] Chat UI at `/`: sleek OpenAI-style, three-dot loader, thinking + done sounds
- [ ] Clickable citations with popup showing highlighted source text
- [ ] Header: "RAG Nutritional Chatbot: Build from Scratch" by Adnan Iqbal Kantroo (romancobblepot)
- [ ] Head metadata + fonts in `__root.tsx`
- [ ] Test API end-to-end, Playwright visual check
- [ ] Publish
