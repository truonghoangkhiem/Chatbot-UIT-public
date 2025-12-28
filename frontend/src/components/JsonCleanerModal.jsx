import React, { useState, useRef } from 'react';

/**
 * JsonCleanerModal - Modal để upload và clean file JSON extraction
 * 
 * Chức năng:
 * - Upload file JSON
 * - Tự động xóa modifications (ảo giác tự sửa đổi)
 * - Phát hiện văn bản gốc vs sửa đổi
 * - Download file đã clean
 */
const JsonCleanerModal = ({ isOpen, onClose }) => {
  const [file, setFile] = useState(null);
  const [jsonData, setJsonData] = useState(null);
  const [cleanedData, setCleanedData] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef(null);

  if (!isOpen) return null;

  // Phát hiện văn bản gốc hay sửa đổi
  const isAmendmentDocument = (data) => {
    const structure = data?.structure || {};
    const doc = structure?.document || {};
    const title = (doc?.title || '').toLowerCase();
    const fullText = (doc?.full_text || '').toLowerCase();
    const sourceFile = (data?.source_file || '').toLowerCase();
    
    const combinedText = `${title} ${fullText} ${sourceFile}`;
    
    const amendmentPatterns = [
      /sửa đổi/i,
      /bổ sung/i,
      /cập nhật/i,
      /thay thế/i,
      /điều chỉnh/i,
      /1393/i,  // Known amendment document
    ];
    
    return amendmentPatterns.some(pattern => pattern.test(combinedText));
  };

  // Kiểm tra self-reference modification
  const isSelfReference = (mod, sourceSignature) => {
    const targetSig = mod?.target_document_signature || '';
    if (!targetSig || !sourceSignature) return false;
    
    const sourceMatch = sourceSignature.match(/(\d+)/);
    const targetMatch = targetSig.match(/(\d+)/);
    
    if (sourceMatch && targetMatch) {
      return sourceMatch[1] === targetMatch[1];
    }
    return false;
  };

  // Trích xuất document signature
  const extractSignature = (data) => {
    const sourceFile = data?.source_file || '';
    const match = sourceFile.match(/(\d+)[_-]?qd[_-]?dhcntt/i);
    if (match) {
      return `${match[1]}/QĐ-ĐHCNTT`;
    }
    
    // Check document title
    const doc = data?.structure?.document || {};
    const title = doc?.title || doc?.full_text || '';
    const sigMatch = title.match(/(\d+)\/QĐ-ĐHCNTT/i);
    if (sigMatch) {
      return `${sigMatch[1]}/QĐ-ĐHCNTT`;
    }
    
    return null;
  };

  // Clean modifications
  const cleanModifications = (data) => {
    const cleaned = JSON.parse(JSON.stringify(data)); // Deep clone
    const signature = extractSignature(data);
    const isOriginal = !isAmendmentDocument(data);
    
    let removedCount = 0;
    
    // Clean structure.articles modifications
    const articles = cleaned?.structure?.articles || [];
    articles.forEach(article => {
      const mods = article?.modifications || [];
      if (mods.length > 0) {
        if (isOriginal) {
          removedCount += mods.length;
          article.modifications = [];
        } else {
          article.modifications = mods.filter(mod => {
            if (isSelfReference(mod, signature)) {
              removedCount++;
              return false;
            }
            if (!mod.target_document_signature) {
              removedCount++;
              return false;
            }
            return true;
          });
        }
      }
    });
    
    // Clean top-level modifications
    if (cleaned.modifications) {
      const mods = cleaned.modifications || [];
      if (isOriginal) {
        removedCount += mods.length;
        cleaned.modifications = [];
      } else {
        cleaned.modifications = mods.filter(mod => {
          if (isSelfReference(mod, signature)) {
            removedCount++;
            return false;
          }
          return true;
        });
      }
    }
    
    // Clean stage2_semantic.modifications
    if (cleaned.stage2_semantic?.modifications) {
      const mods = cleaned.stage2_semantic.modifications || [];
      if (isOriginal) {
        removedCount += mods.length;
        cleaned.stage2_semantic.modifications = [];
      } else {
        cleaned.stage2_semantic.modifications = mods.filter(mod => {
          if (isSelfReference(mod, signature)) {
            removedCount++;
            return false;
          }
          return true;
        });
      }
    }
    
    return {
      cleaned,
      stats: {
        signature,
        isOriginal,
        documentType: isOriginal ? 'Văn bản GỐC' : 'Văn bản SỬA ĐỔI',
        removedCount,
        entitiesCount: cleaned?.stage2_semantic?.entities?.length || 0,
        relationsCount: cleaned?.stage2_semantic?.relations?.length || 0,
        articlesCount: articles.length,
      }
    };
  };

  // Handle file upload
  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    setError(null);
    setFile(selectedFile);
    setCleanedData(null);
    setStats(null);
    setIsProcessing(true);
    
    try {
      const text = await selectedFile.text();
      const data = JSON.parse(text);
      setJsonData(data);
      
      // Auto clean
      const result = cleanModifications(data);
      setCleanedData(result.cleaned);
      setStats(result.stats);
      
    } catch (err) {
      setError(`Lỗi đọc file: ${err.message}`);
      setJsonData(null);
    } finally {
      setIsProcessing(false);
    }
  };

  // Download cleaned file
  const handleDownload = () => {
    if (!cleanedData) return;
    
    const fileName = file?.name?.replace('.json', '_clean.json') || 'cleaned_extraction.json';
    const blob = new Blob([JSON.stringify(cleanedData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Reset
  const handleReset = () => {
    setFile(null);
    setJsonData(null);
    setCleanedData(null);
    setStats(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="w-full max-w-2xl rounded-2xl bg-gray-900 p-6 shadow-2xl border border-gray-700">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              🧹 JSON Cleaner
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Xóa modifications ảo giác từ file extraction
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-gray-400 hover:bg-gray-800 hover:text-white transition-colors"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Upload Area */}
        <div className="mb-6">
          <div 
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors
              ${file ? 'border-green-500 bg-green-500/10' : 'border-gray-600 hover:border-blue-500 hover:bg-blue-500/5'}`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleFileChange}
              className="hidden"
              id="json-file-input"
            />
            <label htmlFor="json-file-input" className="cursor-pointer">
              {file ? (
                <div className="text-green-400">
                  <svg className="h-12 w-12 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="font-medium">{file.name}</p>
                  <p className="text-sm text-gray-400 mt-1">Click để chọn file khác</p>
                </div>
              ) : (
                <div className="text-gray-400">
                  <svg className="h-12 w-12 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="font-medium">Chọn file JSON extraction</p>
                  <p className="text-sm mt-1">Kéo thả hoặc click để upload</p>
                </div>
              )}
            </label>
          </div>
        </div>

        {/* Processing indicator */}
        {isProcessing && (
          <div className="text-center py-4">
            <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
            <p className="text-gray-400 mt-2">Đang xử lý...</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {/* Stats */}
        {stats && (
          <div className="mb-6 space-y-4">
            {/* Document Type Badge */}
            <div className="flex items-center gap-3">
              <span className={`px-3 py-1 rounded-full text-sm font-medium
                ${stats.isOriginal 
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500' 
                  : 'bg-purple-500/20 text-purple-400 border border-purple-500'}`}
              >
                {stats.documentType}
              </span>
              {stats.signature && (
                <span className="text-gray-400 text-sm">
                  📄 {stats.signature}
                </span>
              )}
            </div>
            
            {/* Stats Grid */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold text-red-400">{stats.removedCount}</div>
                <div className="text-sm text-gray-400">Modifications đã xóa</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold text-green-400">{stats.articlesCount}</div>
                <div className="text-sm text-gray-400">Điều khoản</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold text-blue-400">{stats.entitiesCount}</div>
                <div className="text-sm text-gray-400">Thực thể</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold text-yellow-400">{stats.relationsCount}</div>
                <div className="text-sm text-gray-400">Quan hệ</div>
              </div>
            </div>

            {/* Explanation */}
            {stats.isOriginal && stats.removedCount > 0 && (
              <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <p className="text-yellow-400 text-sm">
                  ⚠️ <strong>Đây là văn bản GỐC</strong> - không nên có modifications.
                  Đã xóa {stats.removedCount} modifications ảo giác (self-reference).
                </p>
              </div>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3 justify-end">
          {file && (
            <button
              onClick={handleReset}
              className="px-4 py-2 rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
            >
              🔄 Reset
            </button>
          )}
          
          {cleanedData && (
            <button
              onClick={handleDownload}
              className="px-6 py-2 rounded-lg bg-green-600 text-white hover:bg-green-500 transition-colors font-medium flex items-center gap-2"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download Clean JSON
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default JsonCleanerModal;
