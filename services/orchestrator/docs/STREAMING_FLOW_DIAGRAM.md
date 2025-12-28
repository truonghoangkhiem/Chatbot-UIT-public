# Streaming Response Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                                      │
│  POST /chat                                                                 │
│  { "query": "UIT là gì?", "stream": true, "use_rag": true }               │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API ROUTE: /chat                                    │
│  Check: request.stream == true?                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                        YES ────────┴──────── NO
                         │                    │
                         ▼                    ▼
            ┌────────────────────┐   ┌──────────────────────┐
            │ chat_stream_multi_ │   │ Original Non-Stream  │
            │ agent()            │   │ Logic (ChatResponse) │
            └─────────┬──────────┘   └──────────────────────┘
                      │
                      ▼
            ┌─────────────────────────────────────────┐
            │  STEP 1: PLANNING                       │
            │  SmartPlannerAgent.process()            │
            │  ↓                                       │
            │  SSE: {"type": "planning",              │
            │        "content": "Đang phân tích..."}  │
            └─────────────────┬───────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────────┐
            │  STEP 2: RAG RETRIEVAL                  │
            │  rag_port.retrieve_context()            │
            │  ↓                                       │
            │  SSE: {"type": "status",                │
            │        "content": "Đang tìm kiếm..."}   │
            │  ↓                                       │
            │  SSE: {"type": "status",                │
            │        "content": "Đã tìm thấy 3 docs"} │
            └─────────────────┬───────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────────┐
            │  STEP 3: STREAM ANSWER GENERATION       │
            │  AnswerAgent.stream_process()           │
            │  ↓                                       │
            │  SSE: {"type": "status",                │
            │        "content": "Đang tạo câu trả..."}│
            │  ↓                                       │
            │  ┌──────────────────────────────────┐   │
            │  │ _stream_agent_request()          │   │
            │  │   ↓                               │   │
            │  │ Create AgentRequest(stream=True) │   │
            │  │   ↓                               │   │
            │  │ agent_port.stream_response()     │   │
            │  │   ↓                               │   │
            │  │ OpenRouterAdapter                │   │
            │  │   ↓                               │   │
            │  │ POST to OpenRouter API           │   │
            │  │ (with stream=true)                │   │
            │  └───────────┬──────────────────────┘   │
            │              │                           │
            │              ▼                           │
            │  ┌──────────────────────────────────┐   │
            │  │ LLM STREAMING RESPONSE           │   │
            │  │                                   │   │
            │  │ "UIT" ──→ SSE: {"type":"content",│   │
            │  │               "content":"UIT"}    │   │
            │  │                                   │   │
            │  │ " là" ──→ SSE: {"type":"content",│   │
            │  │               "content":" là"}    │   │
            │  │                                   │   │
            │  │ " viết" ─→ SSE: {"type":"content",│   │
            │  │               "content":" viết"}  │   │
            │  │                                   │   │
            │  │ " tắt" ──→ SSE: {"type":"content",│   │
            │  │               "content":" tắt"}   │   │
            │  │                                   │   │
            │  │ ... (more chunks) ...             │   │
            │  └──────────────────────────────────┘   │
            └─────────────────┬───────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────────┐
            │  STEP 4: COMPLETION                     │
            │  ↓                                       │
            │  SSE: {"type": "done",                  │
            │        "content": "Hoàn thành"}         │
            └─────────────────┬───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLIENT RECEIVES                                     │
│  StreamingResponse (text/event-stream)                                      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │ Browser/Client Display (Progressive)                         │          │
│  ├──────────────────────────────────────────────────────────────┤          │
│  │ [STATUS] Đang phân tích câu hỏi...                           │          │
│  │ [STATUS] Đang tìm kiếm thông tin liên quan...                │          │
│  │ [STATUS] Đã tìm thấy 3 tài liệu liên quan                    │          │
│  │ [STATUS] Đang tạo câu trả lời...                             │          │
│  │                                                               │          │
│  │ UIT là viết tắt của Trường Đại học Công nghệ Thông tin...   │ ◄─── Real-time
│  │                                                               │      updates
│  │ [DONE] Hoàn thành                                            │          │
│  └──────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘


COMPARISON: NON-STREAMING vs STREAMING
═══════════════════════════════════════

NON-STREAMING MODE (stream=false)
──────────────────────────────────
Request → [WAIT] → Full Response
          ▼
    All processing happens
    on server silently
          ▼
    Complete ChatResponse JSON
          ▼
    Display entire response at once


STREAMING MODE (stream=true)
────────────────────────────
Request → Immediate feedback → Progressive display
    ▼            ▼                    ▼
Planning      Status updates      Real-time content
    ▼            ▼                    ▼
RAG Search    Search results      Token by token
    ▼            ▼                    ▼
Generation    Content chunks      Better UX


KEY BENEFITS OF STREAMING
═════════════════════════
✓ Lower perceived latency (content appears immediately)
✓ Better user experience (see progress in real-time)
✓ Handle long responses gracefully (no timeout waiting)
✓ Show intermediate status (user knows what's happening)
✓ Lower memory usage (chunks vs full buffer)
