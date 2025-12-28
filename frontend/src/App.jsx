import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import SettingsModal from './components/SettingsModal';
import SystemInfoModal from './components/SystemInfoModal';
import KGExtractionPanel from './components/KGExtractionPanel';
import JsonCleanerModal from './components/JsonCleanerModal';
import { useChat } from './hooks/useChat';
import { saveToStorage, loadFromStorage } from './utils/helpers';

function App() {
  // Chat management
  const {
    sessionId,
    messages,
    isLoading,
    error,
    ragContext,
    sendMessage,
    newConversation,
    messagesEndRef,
  } = useChat();

  // Settings state - simplified, only UI preferences
  const [settings, setSettings] = useState(() => 
    loadFromStorage('chatbot-settings', {
      showRAGContext: true,  // Only UI setting, no manual parameter overrides
    })
  );

  // UI state
  const [showSettings, setShowSettings] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [showKGExtraction, setShowKGExtraction] = useState(false);
  const [showJsonCleaner, setShowJsonCleaner] = useState(false);
  const [sessions, setSessions] = useState(() => 
    loadFromStorage('chatbot-sessions', [])
  );

  // Save settings when changed
  useEffect(() => {
    saveToStorage('chatbot-settings', settings);
  }, [settings]);

  // Save current session to history
  useEffect(() => {
    if (messages.length > 0) {
      const existingSessionIndex = sessions.findIndex(s => s.id === sessionId);
      const sessionData = {
        id: sessionId,
        title: messages[0]?.content?.substring(0, 50) || 'Cuộc hội thoại mới',
        messages: messages,
        lastUpdated: new Date().toISOString(),
      };

      if (existingSessionIndex >= 0) {
        const updatedSessions = [...sessions];
        updatedSessions[existingSessionIndex] = sessionData;
        setSessions(updatedSessions);
        saveToStorage('chatbot-sessions', updatedSessions);
      } else {
        const updatedSessions = [sessionData, ...sessions];
        setSessions(updatedSessions);
        saveToStorage('chatbot-sessions', updatedSessions);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, sessionId]);

  const handleSendMessage = (message) => {
    // No manual options - SmartPlannerAgent decides everything automatically
    sendMessage(message);
  };

  const handleNewSession = () => {
    newConversation();
  };

  const handleSelectSession = (session) => {
    // This would need to be implemented in useChat hook
    console.log('Load session:', session.id);
  };

  const handleDeleteSession = (sessionId) => {
    const updatedSessions = sessions.filter(s => s.id !== sessionId);
    setSessions(updatedSessions);
    saveToStorage('chatbot-sessions', updatedSessions);
  };

  const handleSaveSettings = (newSettings) => {
    setSettings(newSettings);
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        currentSessionId={sessionId}
        onNewSession={handleNewSession}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onShowSettings={() => setShowSettings(true)}
        onShowInfo={() => setShowInfo(true)}
        onShowKGExtraction={() => setShowKGExtraction(true)}
        onShowJsonCleaner={() => setShowJsonCleaner(true)}
      />

      {/* Main Chat */}
      <div className="flex-1">
        <ChatInterface
          messages={messages}
          isLoading={isLoading}
          onSendMessage={handleSendMessage}
          messagesEndRef={messagesEndRef}
          ragContext={ragContext}
          showRAGContext={settings.showRAGContext}
        />
      </div>

      {/* Modals */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        settings={settings}
        onSave={handleSaveSettings}
      />
      
      <SystemInfoModal
        isOpen={showInfo}
        onClose={() => setShowInfo(false)}
      />

      {/* KG Extraction Panel */}
      {showKGExtraction && (
        <KGExtractionPanel onClose={() => setShowKGExtraction(false)} />
      )}

      {/* JSON Cleaner Modal */}
      <JsonCleanerModal
        isOpen={showJsonCleaner}
        onClose={() => setShowJsonCleaner(false)}
      />

      {/* Error Toast (simple implementation) */}
      {error && (
        <div className="fixed bottom-4 right-4 max-w-sm rounded-lg bg-red-600 p-4 text-white shadow-lg">
          <p className="font-medium">Lỗi</p>
          <p className="text-sm">{error}</p>
        </div>
      )}
    </div>
  );
}

export default App;
