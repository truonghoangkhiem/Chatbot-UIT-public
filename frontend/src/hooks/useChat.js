import { useState, useCallback, useRef, useEffect } from 'react';
import { sendChatMessage } from '../services/api';
import { generateSessionId } from '../utils/helpers';

/**
 * Custom hook for managing chat functionality
 */
export const useChat = (initialSessionId = null) => {
  const [sessionId, setSessionId] = useState(initialSessionId || generateSessionId());
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ragContext, setRagContext] = useState(null);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  /**
   * Send a message to the chatbot
   */
  const sendMessage = useCallback(async (messageText, options = {}) => {
    if (!messageText.trim()) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: messageText,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);
    setRagContext(null);

    try {
      // Send to API
      const response = await sendChatMessage(messageText, sessionId, options);

      // Add bot message
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.response,
        timestamp: response.timestamp,
        metadata: {
          processingStats: response.processing_stats,
          modelUsed: response.model_used,
        },
      };

      setMessages((prev) => [...prev, botMessage]);
      setRagContext(response.rag_context);
    } catch (err) {
      console.error('Error sending message:', err);
      setError(err.response?.data?.detail || 'Đã xảy ra lỗi khi gửi tin nhắn');
      
      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId]);

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
    setRagContext(null);
    setError(null);
  }, []);

  /**
   * Start a new conversation
   */
  const newConversation = useCallback(() => {
    const newSessionId = generateSessionId();
    setSessionId(newSessionId);
    clearMessages();
  }, [clearMessages]);

  /**
   * Load existing conversation
   */
  const loadConversation = useCallback((id, conversationMessages = []) => {
    setSessionId(id);
    setMessages(conversationMessages);
    setRagContext(null);
    setError(null);
  }, []);

  return {
    sessionId,
    messages,
    isLoading,
    error,
    ragContext,
    sendMessage,
    clearMessages,
    newConversation,
    loadConversation,
    messagesEndRef,
  };
};

export default useChat;
