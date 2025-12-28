import React from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot, AlertCircle, Copy, Check } from 'lucide-react';
import { formatTime } from '../utils/helpers';
import { copyToClipboard } from '../utils/helpers';

const MessageBubble = ({ message }) => {
  const [copied, setCopied] = React.useState(false);
  const isUser = message.type === 'user';
  const isError = message.type === 'error';

  const handleCopy = async () => {
    const success = await copyToClipboard(message.content);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''} mb-4`}>
      {/* Avatar */}
      <div
        className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full ${
          isUser
            ? 'bg-blue-600'
            : isError
            ? 'bg-red-600'
            : 'bg-gray-600'
        }`}
      >
        {isUser ? (
          <User className="h-5 w-5 text-white" />
        ) : isError ? (
          <AlertCircle className="h-5 w-5 text-white" />
        ) : (
          <Bot className="h-5 w-5 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} max-w-[80%]`}>
        <div
          className={`rounded-lg px-4 py-2 ${
            isUser
              ? 'bg-blue-600 text-white'
              : isError
              ? 'bg-red-50 text-red-900 border border-red-200'
              : 'bg-gray-100 text-gray-900'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Timestamp and Actions */}
        <div className="mt-1 flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {formatTime(message.timestamp)}
          </span>
          {!isUser && !isError && (
            <button
              onClick={handleCopy}
              className="text-gray-400 hover:text-gray-600 transition-colors"
              title="Copy message"
            >
              {copied ? (
                <Check className="h-3 w-3" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </button>
          )}
        </div>

        {/* Metadata for bot messages */}
        {!isUser && !isError && message.metadata && (
          <div className="mt-1 text-xs text-gray-500">
            {message.metadata.processingStats?.total_time && (
              <span>
                Thời gian xử lý: {message.metadata.processingStats.total_time.toFixed(2)}s
              </span>
            )}
            {message.metadata.modelUsed && (
              <span className="ml-2">Model: {message.metadata.modelUsed}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const MessageList = ({ messages, isLoading, messagesEndRef }) => {
  return (
    <div className="flex-1 overflow-y-auto bg-white p-4">
      {messages.length === 0 ? (
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <Bot className="mx-auto h-16 w-16 text-gray-300 mb-4" />
            <h3 className="text-lg font-semibold text-gray-700 mb-2">
              Chào mừng đến với Chatbot UIT
            </h3>
            <p className="text-gray-500">
              Hãy đặt câu hỏi về UIT, tôi sẽ giúp bạn!
            </p>
          </div>
        </div>
      ) : (
        <>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          
          {/* Typing indicator */}
          {isLoading && (
            <div className="flex gap-3 mb-4">
              <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gray-600">
                <Bot className="h-5 w-5 text-white" />
              </div>
              <div className="flex items-center gap-1 rounded-lg bg-gray-100 px-4 py-3">
                <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </>
      )}
    </div>
  );
};

export default MessageList;
