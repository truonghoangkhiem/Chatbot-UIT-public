import React, { useState } from 'react';
import { MessageSquare, Plus, Trash2, Settings, Info, FileUp, Sparkles } from 'lucide-react';

const Sidebar = ({ 
  sessions = [], 
  currentSessionId, 
  onNewSession, 
  onSelectSession, 
  onDeleteSession,
  onShowSettings,
  onShowInfo,
  onShowKGExtraction,
  onShowJsonCleaner
}) => {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <>
      {/* Mobile Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-4 left-4 z-50 lg:hidden rounded-lg bg-blue-600 p-2 text-white shadow-lg"
      >
        <MessageSquare className="h-6 w-6" />
      </button>

      {/* Sidebar */}
      <div
        className={`fixed lg:static inset-y-0 left-0 z-40 w-64 bg-gray-900 text-white transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Header */}
        <div className="border-b border-gray-800 p-4">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Bot className="h-6 w-6" />
            Chatbot UIT
          </h1>
        </div>

        {/* New Session Button */}
        <div className="p-4">
          <button
            onClick={onNewSession}
            className="flex w-full items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium transition-colors hover:bg-blue-700"
          >
            <Plus className="h-4 w-4" />
            Cuộc hội thoại mới
          </button>
        </div>

        {/* Session List */}
        <div className="flex-1 overflow-y-auto px-4">
          <h2 className="mb-2 text-xs font-semibold uppercase text-gray-400">
            Lịch sử ({sessions.length})
          </h2>
          <div className="space-y-1">
            {sessions.length === 0 ? (
              <p className="py-4 text-center text-sm text-gray-500">
                Chưa có cuộc hội thoại
              </p>
            ) : (
              sessions.map((session) => (
                <div
                  key={session.id}
                  className={`group flex items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors ${
                    session.id === currentSessionId
                      ? 'bg-gray-800'
                      : 'hover:bg-gray-800'
                  }`}
                >
                  <button
                    onClick={() => onSelectSession(session)}
                    className="flex-1 truncate text-left"
                  >
                    {session.title || 'Cuộc hội thoại'}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-400 transition-opacity"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-800 p-4 space-y-2">
          <button
            onClick={onShowKGExtraction}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors hover:bg-gray-800 bg-gradient-to-r from-purple-600/20 to-blue-600/20 border border-purple-500/30"
          >
            <FileUp className="h-4 w-4 text-purple-400" />
            <span className="text-purple-300">Trích xuất KG</span>
          </button>
          <button
            onClick={onShowJsonCleaner}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors hover:bg-gray-800 bg-gradient-to-r from-green-600/20 to-teal-600/20 border border-green-500/30"
          >
            <Sparkles className="h-4 w-4 text-green-400" />
            <span className="text-green-300">JSON Cleaner</span>
          </button>
          <button
            onClick={onShowInfo}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors hover:bg-gray-800"
          >
            <Info className="h-4 w-4" />
            Thông tin hệ thống
          </button>
          <button
            onClick={onShowSettings}
            className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors hover:bg-gray-800"
          >
            <Settings className="h-4 w-4" />
            Cài đặt
          </button>
        </div>
      </div>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          onClick={() => setIsOpen(false)}
          className="fixed inset-0 z-30 bg-black bg-opacity-50 lg:hidden"
        />
      )}
    </>
  );
};

// Fix missing Bot import
import { Bot } from 'lucide-react';

export default Sidebar;
