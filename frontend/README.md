# Chatbot UIT - Frontend

Modern, responsive frontend interface for the Chatbot UIT system built with React, Vite, and Tailwind CSS.

## 🎨 Features

- **Real-time Chat Interface**: Beautiful, responsive chat UI with typing indicators
- **RAG Context Display**: View retrieved documents and their relevance scores
- **Session Management**: Create, switch between, and delete conversation sessions
- **Customizable Settings**: Adjust RAG parameters, temperature, and response length
- **System Monitoring**: View multi-agent system information and health status
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Dark Mode Sidebar**: Professional dark sidebar with clean white chat area
- **Markdown Support**: Bot responses rendered with markdown formatting

## 🚀 Quick Start

### Prerequisites

- Node.js >= 18.x
- npm >= 9.x
- Backend services running (Orchestrator on port 8001)

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

### Configuration

Create `.env` file (or copy from `.env.example`):

```bash
VITE_API_URL=http://localhost:8001/api/v1
```

### Development

```bash
# Start development server
npm run dev

# The app will be available at http://localhost:5173
```

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── ChatInterface.jsx      # Main chat container
│   │   ├── MessageList.jsx        # Message display with bubbles
│   │   ├── MessageInput.jsx       # Input field with send button
│   │   ├── Sidebar.jsx            # Navigation and session list
│   │   ├── RAGContextPanel.jsx    # Document context display
│   │   ├── SettingsModal.jsx      # Settings configuration
│   │   └── SystemInfoModal.jsx    # System information
│   ├── hooks/               # Custom React hooks
│   │   └── useChat.js             # Chat state management
│   ├── services/            # API integration
│   │   └── api.js                 # Backend API calls
│   ├── utils/               # Utility functions
│   │   └── helpers.js             # Helper functions
│   ├── App.jsx             # Main application component
│   ├── main.jsx            # Application entry point
│   └── index.css           # Global styles with Tailwind
├── public/                  # Static assets
├── .env                     # Environment variables
├── .env.example            # Environment template
├── package.json            # Dependencies and scripts
├── tailwind.config.js      # Tailwind CSS configuration
├── vite.config.js          # Vite configuration
└── README.md               # This file
```

## 🎯 Components Overview

### ChatInterface
Main chat container that combines MessageList, MessageInput, and RAGContextPanel.

### MessageList
Displays chat messages with:
- User/bot avatars
- Markdown rendering for bot responses
- Timestamp display
- Copy message functionality
- Typing indicator animation

### MessageInput
Auto-resizing textarea with:
- Send button
- Keyboard shortcuts (Enter to send, Shift+Enter for new line)
- Loading state
- Character counter

### Sidebar
Navigation panel with:
- Session list
- New conversation button
- Settings and system info access
- Mobile-responsive with slide-out menu

### RAGContextPanel
Collapsible panel showing:
- Retrieved documents
- Relevance scores
- Document metadata
- Processing time

### SettingsModal
Configure:
- RAG enable/disable
- Number of documents to retrieve (1-10)
- Temperature (0-2)
- Max tokens (500-4000)
- Show/hide RAG context panel

### SystemInfoModal
Display:
- Service health status
- Multi-agent pipeline information
- Model details
- System capabilities

## 🔧 API Integration

The frontend communicates with the backend through the API service layer:

```javascript
import { sendChatMessage } from './services/api';

// Send a message
const response = await sendChatMessage(
  'Học phí UIT là bao nhiêu?',
  sessionId,
  {
    useRag: true,
    ragTopK: 5,
    temperature: 0.7,
    maxTokens: 2000
  }
);
```

### Available API Functions

- `sendChatMessage(query, sessionId, options)` - Multi-agent chat
- `sendSimpleChatMessage(query, sessionId, options)` - Simple chat
- `checkHealth()` - Health check
- `getConversations()` - List conversations
- `deleteConversation(sessionId)` - Delete session
- `getAgentsInfo()` - Agent system info
- `testAgents()` - Test agents

## 💾 Local Storage

The app uses localStorage to persist:

- **chatbot-settings**: User preferences
  - RAG settings
  - Temperature
  - Max tokens
  - UI preferences

- **chatbot-sessions**: Conversation history
  - Session ID
  - Messages
  - Timestamps

## 🎨 Styling

Built with Tailwind CSS for:
- Utility-first styling
- Responsive design
- Consistent spacing and colors
- Dark mode support

### Color Scheme

- **Primary**: Blue (`blue-600`)
- **Sidebar**: Dark gray (`gray-900`)
- **Background**: Light gray (`gray-50`)
- **Text**: Dark gray (`gray-900`)
- **Borders**: Light gray (`gray-200`)

## 📱 Responsive Design

- **Desktop (lg)**: Full sidebar + chat + RAG panel
- **Tablet (md)**: Collapsible sidebar + chat
- **Mobile (sm)**: Slide-out sidebar, full-width chat

## 🔍 Debugging

The frontend includes console logging for:
- API requests/responses
- State changes
- Errors

Check browser console for detailed logs.

## 🚧 Troubleshooting

### Backend connection failed
- Ensure backend is running on port 8001
- Check `.env` file has correct `VITE_API_URL`
- Verify CORS is enabled on backend

### Messages not sending
- Check browser console for errors
- Verify backend health: http://localhost:8001/api/v1/health
- Check network tab in DevTools

### RAG context not showing
- Enable in Settings modal
- Ensure backend RAG service is running
- Check that `use_rag: true` in API calls

### Styles not working
- Clear browser cache
- Rebuild: `npm run build`
- Check Tailwind configuration

## 📦 Dependencies

### Core
- **React 19**: UI library
- **React DOM 19**: React renderer
- **Vite 6**: Build tool and dev server

### UI & Styling
- **Tailwind CSS 3.4**: Utility-first CSS framework
- **PostCSS 8.4**: CSS processing
- **Autoprefixer 10.4**: CSS vendor prefixes
- **lucide-react**: Beautiful icon library

### Utilities
- **axios 1.6**: HTTP client
- **react-markdown 9.0**: Markdown rendering

## 🎯 Future Enhancements

- [ ] Real-time streaming responses
- [ ] Voice input/output
- [ ] File upload support
- [ ] Multi-language support
- [ ] Chat export/import
- [ ] Advanced search in history
- [ ] User authentication
- [ ] Theme customization
- [ ] PWA support
- [ ] Mobile app

## 📄 License

MIT License

## 👥 Support

For issues or questions:
1. Check the [main README](../README.md)
2. Review [backend documentation](../services/orchestrator/README.md)
3. Check console logs for errors

---

**Built with ❤️ for UIT Community**


## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
