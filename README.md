# Chatbot-UIT

Hệ thống chatbot thông minh cho UIT sử dụng RAG (Retrieval-Augmented Generation) và Multi-Agent Architecture.

## 🚀 Quick Start - Backend

### 1. Setup
```bash
# Tạo conda environment
conda create -n chatbot-UIT python=3.11 -y
conda activate chatbot-UIT

# Cài đặt dependencies
cd services/rag_services && pip install -r requirements.txt
cd ../orchestrator && pip install -r requirements.txt
```

### 2. Khởi động Backend (1 lệnh duy nhất)
```bash
conda activate chatbot-UIT
python start_backend.py
```

**💡 Debug mode được BẬT MẶC ĐỊNH** để hiển thị chi tiết input/output của agents.

Tắt debug mode (production):
```bash
python start_backend.py --no-debug
```

Xem chi tiết trong [BACKEND_SETUP.md](BACKEND_SETUP.md) và [DEBUG_LOGGING_GUIDE.md](DEBUG_LOGGING_GUIDE.md)

### 3. Dừng Backend
```bash
python stop_backend.py
```
hoặc nhấn `Ctrl+C` trong terminal đang chạy.

## 📦 Services

Khi backend chạy, các services sau sẽ khởi động:

- **Orchestrator API**: http://localhost:8001
  - Docs: http://localhost:8001/docs
  - Health: http://localhost:8001/api/v1/health

- **RAG Service**: http://localhost:8000
  - Docs: http://localhost:8000/docs
  - Health: http://localhost:8000/v1/health

- **OpenSearch**: http://localhost:9200
  - Dashboard: http://localhost:5601

- **Weaviate**: http://localhost:8090

## 🏗️ Architecture

```
Frontend (Port 5173) - React + Tailwind CSS
       ↓
Orchestrator (8001) - Điều phối agents
       ↓
RAG Service (8000) - Tìm kiếm tài liệu
       ↓
   ┌────────┬────────┐
Weaviate  OpenSearch
(Vector)  (Keyword)
```

## 🎨 Quick Start - Frontend

### Prerequisites
- Node.js >= 18.x
- Backend services running

### Start Frontend
```bash
cd frontend
npm install
npm run dev
# Hoặc sử dụng script:
./start_frontend.sh
```

Frontend sẽ chạy tại: **http://localhost:5173**

### Features
- ✅ Real-time chat interface
- ✅ RAG context display
- ✅ Session management
- ✅ Customizable settings
- ✅ System monitoring
- ✅ Responsive design
- ✅ Markdown support

Xem thêm trong [Frontend README](frontend/README.md)

## 🎯 Phát triển Full Stack

### 1. Start Backend
```bash
conda activate chatbot-UIT
python start_backend.py
```

### 2. Start Frontend (terminal mới)
```bash
cd frontend
npm run dev
```

### 3. Truy cập
- **Frontend**: http://localhost:5173
- **Backend API Docs**: http://localhost:8001/docs
- **RAG API Docs**: http://localhost:8000/docs

## 📚 Documentation

- [Backend Setup Guide](BACKEND_SETUP.md) - Chi tiết về cài đặt và troubleshooting
- [Orchestrator README](services/orchestrator/README.md) - Agent configuration
- [RAG Service README](services/rag_services/README.md) - Vector & keyword search

## 🧪 Testing

```bash
# Test toàn bộ hệ thống
python services/orchestrator/tests/demo_agent_rag.py

# Test riêng các services
curl http://localhost:8000/v1/health
curl http://localhost:8001/api/v1/health
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Web framework
- **Weaviate** - Vector database
- **OpenSearch** - Keyword search engine
- **LangChain** - LLM orchestration
- **Sentence Transformers** - Embeddings

### Frontend
- **React 19** - UI Library
- **Vite 6** - Build tool
- **Tailwind CSS 3.4** - Styling framework
- **Axios** - HTTP client
- **React Markdown** - Markdown rendering
- **Lucide React** - Icons

## 📁 Project Structure

```
Chatbot-UIT/
├── start_backend.py          # 🚀 Script khởi động backend (MAIN)
├── stop_backend.py            # 🛑 Script dừng backend
├── BACKEND_SETUP.md           # 📖 Hướng dẫn chi tiết
├── services/
│   ├── orchestrator/          # Agent orchestration service
│   │   ├── app/
│   │   ├── config/
│   │   └── tests/
│   └── rag_services/          # RAG search service
│       ├── adapters/
│       ├── core/
│       ├── docker/            # Docker compose files
│       └── retrieval/
└── frontend/                  # 🎨 React + Tailwind CSS UI
    ├── src/
    │   ├── components/        # UI components
    │   ├── services/          # API integration
    │   ├── hooks/             # Custom hooks
    │   └── utils/             # Helper functions
    ├── start_frontend.sh      # Frontend startup script
    └── README.md              # Frontend documentation
```

## 🆘 Troubleshooting

### Backend không khởi động?
```bash
# Check Docker
docker ps

# Check ports
lsof -i :8000,8001

# Check conda env
conda activate chatbot-UIT
python --version  # Should be 3.11.x
```

### Port bị chiếm?
```bash
# Kill process
lsof -ti:8000 | xargs kill -9
lsof -ti:8001 | xargs kill -9
```

Xem thêm trong [BACKEND_SETUP.md](BACKEND_SETUP.md)

## 📝 License

MIT License

## 👥 Contributors

- Backend: Multi-Agent RAG System
- Frontend: [Bạn sẽ phát triển]

