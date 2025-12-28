# Streaming Response Implementation Guide

## Overview

This document explains the streaming response implementation for the Chatbot-UIT orchestrator service. The streaming feature allows real-time response generation, providing better user experience for long-running queries.

## Architecture

### Components Modified

1. **AnswerAgent** (`services/orchestrator/app/agents/answer_agent.py`)
   - Added `stream_process()` method for streaming answer generation
   - Added `_stream_agent_request()` helper method for streaming LLM requests

2. **API Routes** (`services/orchestrator/app/api/routes.py`)
   - Modified `/chat` endpoint to support both streaming and non-streaming
   - Added `chat_stream_multi_agent()` function for streaming orchestration
   - Returns `StreamingResponse` with Server-Sent Events (SSE) format

### Flow Diagram

```
Client Request (stream=true)
    ↓
/chat endpoint
    ↓
Check if streaming enabled?
    ↓ YES
chat_stream_multi_agent()
    ↓
1. Planning (SmartPlannerAgent)
    → Send status: "Đang phân tích câu hỏi..."
    ↓
2. RAG Retrieval
    → Send status: "Đang tìm kiếm thông tin liên quan..."
    → Send status: "Đã tìm thấy X tài liệu"
    ↓
3. Stream Answer Generation (AnswerAgent.stream_process)
    → Send status: "Đang tạo câu trả lời..."
    → Stream content chunks in real-time
    ↓
4. Send completion
    → Send: "Hoàn thành"
```

## Implementation Details

### 1. AnswerAgent Streaming

**New Method: `stream_process()`**

```python
async def stream_process(self, input_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """Stream comprehensive answer generation from context and query."""
    query = input_data.get("query", "")
    context_documents = input_data.get("context_documents", [])
    rewritten_queries = input_data.get("rewritten_queries", [])
    previous_context = input_data.get("previous_context", "")
    previous_feedback = input_data.get("previous_feedback", "")
    
    # Build the answer generation prompt
    prompt = self._build_answer_prompt(
        query, context_documents, rewritten_queries, previous_context, previous_feedback
    )
    
    # Stream response from the agent
    async for chunk in self._stream_agent_request(prompt):
        yield chunk
```

**Key Features:**
- Reuses existing `_build_answer_prompt()` method for consistency
- Streams chunks directly from LLM through `_stream_agent_request()`
- Maintains same prompt structure as non-streaming mode

### 2. API Endpoint Streaming

**Modified: `/chat` Endpoint**

```python
async def chat(request: ChatRequest):
    # If streaming is requested, use the streaming endpoint
    if request.stream:
        return await chat_stream_multi_agent(request)
    
    # ... existing non-streaming logic
```

**New: `chat_stream_multi_agent()` Function**

Streams responses in Server-Sent Events (SSE) format:

```python
async def generate_stream() -> AsyncGenerator[str, None]:
    # Step 1: Planning
    yield f"data: {json.dumps({'type': 'planning', 'content': '...'})}\n\n"
    
    # Step 2: RAG Retrieval
    yield f"data: {json.dumps({'type': 'status', 'content': '...'})}\n\n"
    
    # Step 3: Stream answer
    async for chunk in answer_agent.stream_process(answer_input):
        yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
    
    # Step 4: Completion
    yield f"data: {json.dumps({'type': 'done', 'content': 'Hoàn thành'})}\n\n"
```

### 3. Server-Sent Events (SSE) Format

**Event Types:**

| Type | Description | Example |
|------|-------------|---------|
| `planning` | Planning phase started | `{'type': 'planning', 'content': 'Đang phân tích câu hỏi...'}` |
| `status` | Status update | `{'type': 'status', 'content': 'Đang tìm kiếm thông tin...'}` |
| `warning` | Non-fatal warning | `{'type': 'warning', 'content': 'Không tìm thấy tài liệu...'}` |
| `content` | Actual answer content | `{'type': 'content', 'content': 'UIT là...'}` |
| `done` | Streaming completed | `{'type': 'done', 'content': 'Hoàn thành'}` |
| `error` | Error occurred | `{'type': 'error', 'content': 'Đã có lỗi...'}` |

**SSE Message Format:**

```
data: {"type": "content", "content": "chunk of text"}\n\n
```

## Usage

### API Request

**Streaming Enabled:**

```json
{
  "query": "UIT là gì?",
  "session_id": "session_123",
  "use_rag": true,
  "rag_top_k": 3,
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**Non-Streaming (Traditional):**

```json
{
  "query": "UIT là gì?",
  "session_id": "session_123",
  "use_rag": true,
  "rag_top_k": 3,
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### Client Implementation

#### Python with aiohttp

```python
async with aiohttp.ClientSession() as session:
    async with session.post(url, json=request_data) as response:
        async for line in response.content:
            line_str = line.decode('utf-8').strip()
            
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                data = json.loads(data_str)
                
                if data['type'] == 'content':
                    print(data['content'], end='', flush=True)
                elif data['type'] == 'done':
                    print("\n\nCompleted!")
                    break
```

#### JavaScript/TypeScript with fetch

```javascript
const response = await fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(requestData)
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      
      if (data.type === 'content') {
        // Append to UI
        displayContent(data.content);
      } else if (data.type === 'done') {
        console.log('Streaming completed');
      }
    }
  }
}
```

#### React Hook Example

```javascript
const useStreamingChat = () => {
  const [content, setContent] = useState('');
  const [status, setStatus] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);

  const streamChat = async (query) => {
    setIsStreaming(true);
    setContent('');
    
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, stream: true })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          
          if (data.type === 'content') {
            setContent(prev => prev + data.content);
          } else if (data.type === 'status') {
            setStatus(data.content);
          } else if (data.type === 'done') {
            setIsStreaming(false);
          }
        }
      }
    }
  };

  return { content, status, isStreaming, streamChat };
};
```

## Testing

### Run Test Script

```bash
# Make sure orchestrator service is running
cd services/orchestrator
python app/main.py

# In another terminal, run the test
cd services/orchestrator
python test_streaming.py
```

### Manual Testing with curl

```bash
# Streaming request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "UIT là gì?",
    "stream": true,
    "use_rag": true,
    "rag_top_k": 3
  }' \
  --no-buffer

# Non-streaming request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "UIT là gì?",
    "stream": false,
    "use_rag": true,
    "rag_top_k": 3
  }'
```

## Performance Considerations

### Streaming Benefits

1. **Improved User Experience**
   - Users see content immediately as it's generated
   - Reduces perceived latency
   - Better for long responses

2. **Lower Memory Usage**
   - Content is sent in chunks, not buffered entirely
   - Reduces server memory footprint

3. **Better Error Handling**
   - Can send partial content even if generation fails midway
   - Users get immediate feedback on progress

### Trade-offs

1. **No Response Validation**
   - Cannot verify complete response before sending
   - ResponseFormatter step is skipped in streaming mode

2. **Limited Retry Capability**
   - Cannot easily retry if quality is poor
   - Feedback loop mechanism not applicable

3. **Client Complexity**
   - Clients must handle streaming protocol
   - More complex error handling

## Configuration

### Response Headers

The streaming endpoint sets these headers:

```python
{
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"  # Disable nginx buffering
}
```

### Timeouts

- Default timeout: 30 seconds (configurable in AgentConfig)
- Streaming maintains connection until completion or error
- Client should implement connection timeout

## Error Handling

### Server-Side Errors

```python
try:
    async for chunk in answer_agent.stream_process(answer_input):
        yield chunk
except Exception as e:
    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
```

### Client-Side Handling

```javascript
if (data.type === 'error') {
  console.error('Error occurred:', data.content);
  setError(data.content);
  setIsStreaming(false);
}
```

## Future Enhancements

1. **Progressive RAG Retrieval**
   - Stream RAG results as they become available
   - Show citations in real-time

2. **ResponseFormatter Integration**
   - Implement streaming verification
   - Progressive quality checks during generation

3. **Token Usage Tracking**
   - Send token count updates during streaming
   - Real-time cost estimation

4. **Cancellation Support**
   - Allow clients to cancel ongoing streams
   - Clean up resources properly

5. **Resume Capability**
   - Support resuming interrupted streams
   - Session-based stream recovery

## Summary

The streaming implementation provides:

✅ Real-time response generation  
✅ Better user experience for long responses  
✅ Server-Sent Events (SSE) protocol  
✅ Backward compatible with non-streaming mode  
✅ Multi-agent orchestration support  
✅ Comprehensive error handling  

The implementation maintains consistency with the existing architecture while adding streaming capabilities for improved user experience.
