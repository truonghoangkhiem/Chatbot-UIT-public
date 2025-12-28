import React, { useState } from 'react';
import { X, Save, Info } from 'lucide-react';

const SettingsModal = ({ isOpen, onClose, settings, onSave }) => {
  const [localSettings, setLocalSettings] = useState(settings);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(localSettings);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="w-full max-w-md rounded-lg bg-white shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-200 p-4">
          <h2 className="text-xl font-semibold">Cài đặt giao diện</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Info about automatic system */}
          <div className="rounded-lg bg-blue-50 p-3 border border-blue-200">
            <div className="flex items-start gap-2">
              <Info className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-blue-900">Hệ thống tự động</p>
                <p className="text-xs text-blue-700 mt-1">
                  Chatbot sử dụng SmartPlannerAgent để tự động quyết định:
                </p>
                <ul className="text-xs text-blue-700 mt-1 ml-4 list-disc">
                  <li>Có cần tìm kiếm tài liệu (RAG) hay không</li>
                  <li>Sử dụng Knowledge Graph hay Vector Search</li>
                  <li>Các siêu tham số tối ưu cho từng câu hỏi</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Show RAG Context */}
          <div className="pt-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={localSettings.showRAGContext}
                onChange={(e) =>
                  setLocalSettings({ ...localSettings, showRAGContext: e.target.checked })
                }
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm font-medium">Hiển thị ngữ cảnh truy xuất</span>
            </label>
            <p className="ml-6 text-xs text-gray-500 mt-1">
              Hiển thị panel bên phải với các tài liệu được hệ thống sử dụng để trả lời
            </p>
          </div>

          {/* Pipeline Info */}
          <div className="rounded-lg bg-gray-50 p-3 border border-gray-200">
            <p className="text-sm font-medium text-gray-900 mb-2">Pipeline đang sử dụng</p>
            <div className="flex items-center gap-2 text-xs text-gray-600">
              <span className="px-2 py-1 bg-green-100 text-green-700 rounded">SmartPlanner</span>
              <span>→</span>
              <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded">AnswerAgent</span>
              <span>→</span>
              <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded">ResponseFormatter</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              3-Agent Pipeline tối ưu (tiết kiệm 40% chi phí LLM)
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 border-t border-gray-200 p-4">
          <button
            onClick={onClose}
            className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium hover:bg-gray-50"
          >
            Hủy
          </button>
          <button
            onClick={handleSave}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
          >
            <Save className="h-4 w-4" />
            Lưu
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
