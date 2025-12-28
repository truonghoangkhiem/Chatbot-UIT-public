# Chatbot-UIT Orchestrator Service

## Tổng quan

Orchestrator Service là thành phần trung tâm của hệ thống Chatbot-UIT, chịu trách nhiệm điều phối giữa việc truy xuất thông tin (RAG) và sinh phản hồi từ AI agent. Service này được thiết kế theo kiến trúc **Ports & Adapters** để đảm bảo tính linh hoạt và khả năng mở rộng.

## Kiến trúc

```
orchestrator/
├── app/
│   ├── core/                          # Domain layer
│   │   ├── domain.py                  # Domain models
│   │   ├── orchestration_service.py   # Core business logic
│   │   └── container.py              # Dependency injection
│   ├── ports/                        # Port interfaces
│   │   └── agent_ports.py            # Service contracts
│   ├── adapters/                     # Infrastructure layer
│   │   ├── openrouter_adapter.py     # OpenRouter API integration
│   │   ├── rag_adapter.py            # RAG service integration
│   │   └── conversation_manager.py   # Conversation management
│   ├── api/                          # API layer
│   │   └── routes.py                 # FastAPI endpoints
│   ├── schemas/                      # API schemas
│   │   └── api_schemas.py            # Request/response models
│   └── main.py                       # FastAPI application
├── requirements.txt                  # Dependencies
├── .env.example                     # Environment variables template
├── start_server.sh                  # Server startup script
├── demo_orchestrator.py             # Demo script
└── README.md                        # This file
```

## Tính năng chính

### 1. Orchestration Pipeline
- **RAG Integration**: Truy xuất thông tin liên quan từ RAG service
- **Agent Communication**: Tích hợp với OpenRouter API cho các LLM
- **Context Management**: Quản lý context cuộc hội thoại
- **Response Synthesis**: Kết hợp thông tin RAG với khả năng sinh text của AI

### 2. API Endpoints

#### `/api/v1/chat` (POST)
Endpoint chính để xử lý cuộc hội thoại:
```json
{
  "query": "Hướng dẫn đăng ký học phần tại UIT như thế nào?",
  "session_id": "user_123",
  "use_rag": true,
  "rag_top_k": 5,
  "model": "openai/gpt-3.5-turbo",
  "temperature": 0.7
}
```

#### `/api/v1/chat/stream` (POST)
Streaming response cho real-time chat.

#### `/api/v1/health` (GET)
Health check cho tất cả các service components.

#### `/api/v1/conversations` (GET)
Quản lý các session hội thoại active.

### 3. Kiến trúc Ports & Adapters

#### Ports (Interfaces)
- **AgentPort**: Interface cho LLM agents
- **RAGServicePort**: Interface cho RAG service
- **ConversationManagerPort**: Interface cho conversation management

#### Adapters (Implementations)
- **OpenRouterAdapter**: Tích hợp OpenRouter API
- **RAGServiceAdapter**: Kết nối với RAG service
- **InMemoryConversationManager**: Quản lý conversation in-memory

## Cài đặt và Chạy

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Cấu hình environment
```bash
cp .env.example .env
# Chỉnh sửa .env với cấu hình của bạn
```

### 3. Khởi động service
```bash
./start_server.sh
```

Hoặc manual:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

### 4. Test integration
```bash
python test_basic.py
```

### 5. Demo đầy đủ
```bash
python demo_orchestrator.py
```

## Cấu hình

### Environment Variables

| Variable | Mô tả | Default |
|----------|-------|---------|
| `OPENROUTER_API_KEY` | API key cho OpenRouter | **Required** |
| `OPENROUTER_BASE_URL` | Base URL OpenRouter API | `https://openrouter.ai/api/v1` |
| `OPENROUTER_DEFAULT_MODEL` | Model mặc định | `openai/gpt-3.5-turbo` |
| `RAG_SERVICE_URL` | URL của RAG service | `http://localhost:8001` |
| `DEFAULT_SYSTEM_PROMPT` | System prompt mặc định | Auto-generated |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8002` |

### Supported Models (OpenRouter)
- `openai/gpt-4`
- `openai/gpt-4-turbo`
- `openai/gpt-3.5-turbo`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-sonnet`
- `google/gemini-pro`
- `meta-llama/llama-2-70b-chat`

## API Documentation

Khi service đang chạy, truy cập:
- **Swagger UI**: `http://localhost:8002/docs`
- **ReDoc**: `http://localhost:8002/redoc`

## Ví dụ sử dụng

### 1. Chat đơn giản
```python
import aiohttp

async def simple_chat():
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": "Xin chào!",
            "use_rag": False
        }
        
        async with session.post(
            "http://localhost:8002/api/v1/chat",
            json=payload
        ) as response:
            data = await response.json()
            print(data["response"])
```

### 2. Chat với RAG
```python
payload = {
    "query": "Học phí tại UIT như thế nào?",
    "use_rag": True,
    "rag_top_k": 3,
    "session_id": "user_session_1"
}
```

### 3. Streaming chat
```python
async def stream_chat():
    payload = {
        "query": "Giải thích về quy trình nhập học",
        "stream": True,
        "use_rag": True
    }
    
    async with session.post(
        "http://localhost:8002/api/v1/chat/stream",
        json=payload
    ) as response:
        async for line in response.content:
            # Process streaming data
            pass
```

## Kiến trúc và Nguyên tắc

### 1. Ports & Adapters Pattern
- **Tách biệt rõ ràng** giữa business logic và infrastructure
- **Dependency Injection** để quản lý dependencies
- **Interface-based design** cho flexibility

### 2. Clean Architecture Layers
```
API Layer (FastAPI) 
    ↓
Application Layer (Routes)
    ↓  
Domain Layer (Business Logic)
    ↓
Infrastructure Layer (Adapters)
```

### 3. Error Handling
- Graceful degradation khi RAG service không available
- Retry logic với exponential backoff
- Comprehensive error logging

### 4. Performance Considerations
- Async/await throughout
- Connection pooling
- Streaming support cho large responses

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python demo_orchestrator.py
```

### Manual Testing
```bash
curl -X POST "http://localhost:8002/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Test message",
    "use_rag": false
  }'
```

## Monitoring và Logging

### Health Check
```bash
curl http://localhost:8002/api/v1/health
```

### Logs
Service sử dụng Python logging với các levels:
- INFO: General information
- WARNING: Potential issues
- ERROR: Actual errors
- DEBUG: Detailed debugging info

## Phát triển và Mở rộng

### Thêm Agent Provider mới
1. Implement `AgentPort` interface
2. Tạo adapter class mới
3. Update `ServiceContainer`
4. Thêm cấu hình environment

### Thêm Conversation Storage
1. Implement `ConversationManagerPort`
2. Tạo adapter cho database/Redis
3. Update dependency injection

### Custom Business Logic
Modify `OrchestrationService` để thêm:
- Custom prompt engineering
- Result post-processing
- Advanced conversation flow

## Troubleshooting

### Common Issues

1. **OPENROUTER_API_KEY not set**
   - Set environment variable in `.env`

2. **RAG service connection failed**
   - Check `RAG_SERVICE_URL`
   - Ensure RAG service is running

3. **Import errors**
   - Check Python path
   - Install requirements.txt

4. **Port already in use**
   - Change PORT in `.env`
   - Kill existing process

### Debug Mode
```bash
LOG_LEVEL=DEBUG uvicorn app.main:app --reload
```

## Đóng góp

1. Fork repository
2. Tạo feature branch
3. Implement changes theo Ports & Adapters pattern
4. Add tests
5. Submit pull request

## License

MIT License - Xem file LICENSE để biết thêm chi tiết.