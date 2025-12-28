import React from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import RAGContextPanel from './RAGContextPanel';

const ChatInterface = ({ 
  messages, 
  isLoading, 
  onSendMessage, 
  messagesEndRef,
  ragContext,
  showRAGContext = true
}) => {
  return (
    <div className="flex h-screen flex-col">
      {/* Main Chat Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Messages */}
        <div className="flex flex-1 flex-col">
          <MessageList 
            messages={messages} 
            isLoading={isLoading} 
            messagesEndRef={messagesEndRef}
          />
          <MessageInput 
            onSendMessage={onSendMessage} 
            isLoading={isLoading}
          />
        </div>

        {/* RAG Context Panel */}
        {showRAGContext && ragContext && (
          <RAGContextPanel ragContext={ragContext} />
        )}
      </div>
    </div>
  );
};

export default ChatInterface;
