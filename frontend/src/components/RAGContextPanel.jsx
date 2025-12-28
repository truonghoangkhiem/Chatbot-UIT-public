import React, { useState } from 'react';
import { FileText, X, ChevronDown, ChevronUp, Database, Search, Clock, Brain, GitBranch, Box } from 'lucide-react';
import { formatProcessingTime } from '../utils/helpers';

const RAGContextPanel = ({ ragContext }) => {
  const [isOpen, setIsOpen] = useState(true);
  const [expandedDocs, setExpandedDocs] = useState(new Set());

  if (!ragContext) {
    return (
      <div className={`border-l border-gray-200 bg-gray-50 transition-all duration-300 ${
        isOpen ? 'w-80' : 'w-12'
      }`}>
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-200 bg-white p-4">
          {isOpen ? (
            <>
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-gray-400" />
                <h3 className="font-semibold text-gray-500">Ngữ cảnh truy xuất</h3>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="h-5 w-5" />
              </button>
            </>
          ) : (
            <button
              onClick={() => setIsOpen(true)}
              className="mx-auto text-gray-400 hover:text-gray-600"
            >
              <FileText className="h-5 w-5" />
            </button>
          )}
        </div>
        {isOpen && (
          <div className="p-4 text-center text-gray-500 text-sm">
            <Brain className="h-12 w-12 mx-auto mb-2 text-gray-300" />
            <p>Gửi câu hỏi để xem ngữ cảnh được truy xuất</p>
          </div>
        )}
      </div>
    );
  }

  const hasDocuments = ragContext.documents && ragContext.documents.length > 0;

  const toggleDoc = (index) => {
    const newExpanded = new Set(expandedDocs);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedDocs(newExpanded);
  };

  // Determine complexity display
  const getComplexityInfo = (complexity) => {
    const levels = {
      'simple': { label: 'Đơn giản', color: 'green', bg: 'bg-green-100', text: 'text-green-700' },
      'medium': { label: 'Trung bình', color: 'yellow', bg: 'bg-yellow-100', text: 'text-yellow-700' },
      'complex': { label: 'Phức tạp', color: 'red', bg: 'bg-red-100', text: 'text-red-700' },
    };
    return levels[complexity?.toLowerCase()] || { label: complexity || 'N/A', color: 'gray', bg: 'bg-gray-100', text: 'text-gray-600' };
  };

  const complexityInfo = getComplexityInfo(ragContext.complexity);

  return (
    <div className={`border-l border-gray-200 bg-gray-50 transition-all duration-300 ${
      isOpen ? 'w-96' : 'w-12'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 bg-white p-4">
        {isOpen ? (
          <>
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-blue-600" />
              <h3 className="font-semibold">Ngữ cảnh truy xuất</h3>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </>
        ) : (
          <button
            onClick={() => setIsOpen(true)}
            className="mx-auto text-gray-400 hover:text-gray-600"
          >
            <FileText className="h-5 w-5" />
          </button>
        )}
      </div>

      {/* Content */}
      {isOpen && (
        <div className="overflow-y-auto p-4" style={{ maxHeight: 'calc(100vh - 64px)' }}>
          {/* Search Sources Indicators */}
          <div className="mb-4 space-y-2">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
              Nguồn tìm kiếm
            </p>
            
            {/* Search Source Badges */}
            <div className="flex gap-2 flex-wrap">
              {/* Knowledge Graph Badge */}
              <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border ${
                ragContext.use_knowledge_graph 
                  ? 'bg-purple-100 border-purple-300 text-purple-700' 
                  : 'bg-gray-100 border-gray-200 text-gray-400'
              }`}>
                <GitBranch className="h-4 w-4" />
                <span className="text-xs font-medium">Knowledge Graph</span>
                {ragContext.use_knowledge_graph ? (
                  <span className="text-xs">✓</span>
                ) : (
                  <span className="text-xs">✗</span>
                )}
              </div>
              
              {/* Vector Search Badge */}
              <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border ${
                ragContext.use_vector_search !== false
                  ? 'bg-blue-100 border-blue-300 text-blue-700' 
                  : 'bg-gray-100 border-gray-200 text-gray-400'
              }`}>
                <Box className="h-4 w-4" />
                <span className="text-xs font-medium">Vector Search</span>
                {ragContext.use_vector_search !== false ? (
                  <span className="text-xs">✓</span>
                ) : (
                  <span className="text-xs">✗</span>
                )}
              </div>
            </div>

            {/* Complexity & Strategy Info */}
            {(ragContext.complexity || ragContext.strategy) && (
              <div className="rounded-lg bg-white p-3 border border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Brain className="h-4 w-4 text-gray-500" />
                    <span className="text-xs text-gray-500">SmartPlanner Analysis</span>
                  </div>
                  {ragContext.complexity && (
                    <span className={`text-xs px-2 py-0.5 rounded ${complexityInfo.bg} ${complexityInfo.text}`}>
                      {complexityInfo.label}
                    </span>
                  )}
                </div>
                {ragContext.strategy && (
                  <p className="text-xs text-gray-600 mt-1 ml-6">
                    Strategy: <span className="font-medium">{ragContext.strategy}</span>
                  </p>
                )}
              </div>
            )}

            {/* Stats */}
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded-lg bg-white p-2 border border-gray-200">
                <div className="flex items-center gap-1 text-gray-500 text-xs mb-1">
                  <Database className="h-3 w-3" />
                  <span>Tài liệu</span>
                </div>
                <p className="font-semibold text-gray-900">
                  {ragContext.total_documents || ragContext.documents?.length || 0}
                </p>
              </div>
              <div className="rounded-lg bg-white p-2 border border-gray-200">
                <div className="flex items-center gap-1 text-gray-500 text-xs mb-1">
                  <Clock className="h-3 w-3" />
                  <span>Thời gian</span>
                </div>
                <p className="font-semibold text-gray-900">
                  {ragContext.processing_time 
                    ? formatProcessingTime(ragContext.processing_time)
                    : 'N/A'}
                </p>
              </div>
            </div>

            {/* Rewritten Query if available */}
            {ragContext.rewritten_query && (
              <div className="rounded-lg bg-yellow-50 p-3 border border-yellow-200">
                <div className="flex items-center gap-1 text-yellow-700 text-xs mb-1">
                  <Search className="h-3 w-3" />
                  <span className="font-medium">Query đã tối ưu</span>
                </div>
                <p className="text-sm text-yellow-900 italic">
                  "{ragContext.rewritten_query}"
                </p>
              </div>
            )}
          </div>

          {/* Documents */}
          {hasDocuments ? (
            <div className="space-y-3">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                Tài liệu tham khảo
              </p>
              {ragContext.documents.map((doc, index) => (
                <div
                  key={index}
                  className="rounded-lg border border-gray-200 bg-white overflow-hidden"
                >
                  <button
                    onClick={() => toggleDoc(index)}
                    className="flex w-full items-center justify-between p-3 text-left hover:bg-gray-50"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-100 text-blue-600 text-xs flex items-center justify-center font-medium">
                          {index + 1}
                        </span>
                        <h4 className="font-medium text-sm text-gray-900 line-clamp-2">
                          {doc.title || `Tài liệu ${index + 1}`}
                        </h4>
                      </div>
                      <div className="flex items-center gap-2 mt-1 ml-7">
                        <span className={`text-xs px-1.5 py-0.5 rounded ${
                          doc.score >= 0.8 ? 'bg-green-100 text-green-700' :
                          doc.score >= 0.5 ? 'bg-yellow-100 text-yellow-700' :
                          'bg-gray-100 text-gray-600'
                        }`}>
                          {(doc.score * 100).toFixed(0)}% liên quan
                        </span>
                        {doc.source && (
                          <span className="text-xs text-gray-400">
                            {doc.source}
                          </span>
                        )}
                      </div>
                    </div>
                    {expandedDocs.has(index) ? (
                      <ChevronUp className="h-4 w-4 text-gray-400 flex-shrink-0 ml-2" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-gray-400 flex-shrink-0 ml-2" />
                    )}
                  </button>
                  
                  {expandedDocs.has(index) && (
                    <div className="border-t border-gray-200 p-3 bg-gray-50">
                      <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                        {doc.content}
                      </p>
                      {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                        <div className="mt-3 pt-2 border-t border-gray-200">
                          <p className="text-xs text-gray-500 font-medium mb-1">Metadata:</p>
                          <div className="flex flex-wrap gap-1">
                            {Object.entries(doc.metadata).map(([key, value]) => (
                              <span key={key} className="text-xs bg-gray-100 px-2 py-0.5 rounded">
                                {key}: {String(value).substring(0, 50)}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4">
              <Brain className="h-10 w-10 mx-auto mb-2 text-gray-300" />
              <p className="text-sm text-gray-500">
                SmartPlanner quyết định không cần RAG cho câu hỏi này
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RAGContextPanel;
