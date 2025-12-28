import React, { useState, useEffect, useRef } from 'react';

const RAG_SERVICE_URL = import.meta.env.VITE_RAG_SERVICE_URL || 'http://localhost:8002';

/**
 * Knowledge Graph Extraction Component
 * 
 * Two-Stage Pipeline:
 * - Stage 1: Upload PDF → VLM extracts structure → Download JSON
 * - Stage 2: Upload Stage 1 JSON → LLM extracts semantics → Merged result
 */
const KGExtractionPanel = ({ onClose }) => {
  // Mode: 'stage1' | 'stage2' | 'full'
  const [mode, setMode] = useState('stage1');
  
  // Common states
  const [file, setFile] = useState(null);
  const [category, setCategory] = useState('Quy chế Đào tạo');
  const [pushToNeo4j, setPushToNeo4j] = useState(false);
  const [clearExisting, setClearExisting] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [neo4jStats, setNeo4jStats] = useState(null);
  
  const fileInputRef = useRef(null);
  const pollingRef = useRef(null);

  // Poll for status updates
  useEffect(() => {
    if (jobId && status?.status !== 'completed' && status?.status !== 'failed') {
      pollingRef.current = setInterval(async () => {
        try {
          const response = await fetch(`${RAG_SERVICE_URL}/v1/extraction/status/${jobId}`);
          const data = await response.json();
          setStatus(data);
          
          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(pollingRef.current);
            
            if (data.status === 'completed') {
              // For index modes, just use the stats from status (no result file)
              if (['neo4j', 'weaviate', 'opensearch'].includes(mode)) {
                setResult({
                  message: `Index ${mode} thành công!`,
                  stats: data.stats
                });
              } else {
                // For extraction modes, fetch the result file
                const resultResponse = await fetch(`${RAG_SERVICE_URL}/v1/extraction/result/${jobId}`);
                const resultData = await resultResponse.json();
                setResult(resultData);
              }
            }
          }
        } catch (err) {
          console.error('Error polling status:', err);
        }
      }, 1000);
    }
    
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [jobId, status?.status, mode]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    const jsonModes = ['stage2', 'neo4j', 'weaviate', 'opensearch'];
    const isJsonMode = jsonModes.includes(mode);
    
    if (selectedFile) {
      if (isJsonMode && selectedFile.name.endsWith('.json')) {
        setFile(selectedFile);
        setError(null);
      } else if (!isJsonMode && selectedFile.type === 'application/pdf') {
        setFile(selectedFile);
        setError(null);
      } else {
        setError(`Vui lòng chọn file ${isJsonMode ? 'JSON' : 'PDF'}`);
      }
    }
  };

  const handleUpload = async () => {
    const jsonModes = ['stage2', 'neo4j', 'weaviate', 'opensearch'];
    const isJsonMode = jsonModes.includes(mode);
    
    if (!file) {
      setError(isJsonMode ? 'Vui lòng chọn file JSON' : 'Vui lòng chọn file PDF');
      return;
    }

    setIsUploading(true);
    setError(null);
    setResult(null);
    setStatus(null);

    try {
      let endpoint = '';
      let formData = new FormData();
      
      if (mode === 'stage1') {
        endpoint = `${RAG_SERVICE_URL}/v1/extraction/stage1/upload`;
        formData.append('file', file);
        const url = new URL(endpoint);
        url.searchParams.append('category', category);
        
        const response = await fetch(url, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Upload failed');
        }
        
        const data = await response.json();
        setJobId(data.job_id);
        setStatus(data);
        
      } else if (mode === 'stage2') {
        endpoint = `${RAG_SERVICE_URL}/v1/extraction/stage2/upload`;
        formData.append('file', file);
        const url = new URL(endpoint);
        url.searchParams.append('category', category);
        url.searchParams.append('push_to_neo4j', pushToNeo4j);
        
        const response = await fetch(url, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Upload failed');
        }
        
        const data = await response.json();
        setJobId(data.job_id);
        setStatus(data);
        
      } else if (mode === 'neo4j') {
        // Neo4j Import mode
        endpoint = `${RAG_SERVICE_URL}/v1/extraction/neo4j/upload`;
        formData.append('file', file);
        const url = new URL(endpoint);
        url.searchParams.append('clear_existing', clearExisting);
        
        const response = await fetch(url, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Upload failed');
        }
        
        const data = await response.json();
        setJobId(data.job_id);
        setStatus(data);
        
      } else if (mode === 'weaviate') {
        // Weaviate Vector Index mode
        endpoint = `${RAG_SERVICE_URL}/v1/extraction/weaviate/upload`;
        formData.append('file', file);
        const url = new URL(endpoint);
        url.searchParams.append('doc_type', category);
        
        const response = await fetch(url, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Upload failed');
        }
        
        const data = await response.json();
        setJobId(data.job_id);
        setStatus(data);
        
      } else if (mode === 'opensearch') {
        // OpenSearch BM25 Index mode
        endpoint = `${RAG_SERVICE_URL}/v1/extraction/opensearch/upload`;
        formData.append('file', file);
        const url = new URL(endpoint);
        url.searchParams.append('doc_type', category);
        url.searchParams.append('clear_existing', clearExisting);
        
        const response = await fetch(url, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Upload failed');
        }
        
        const data = await response.json();
        setJobId(data.job_id);
        setStatus(data);
        
      } else {
        // Full pipeline
        endpoint = `${RAG_SERVICE_URL}/v1/extraction/upload`;
        formData.append('file', file);
        const url = new URL(endpoint);
        url.searchParams.append('category', category);
        url.searchParams.append('push_to_neo4j', pushToNeo4j);
        
        const response = await fetch(url, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Upload failed');
        }
        
        const data = await response.json();
        setJobId(data.job_id);
        setStatus(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDownload = () => {
    if (result) {
      const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const prefix = mode === 'stage1' ? 'stage1' : mode === 'stage2' ? 'merged' : 'extraction';
      a.download = `${prefix}_${jobId.slice(0, 8)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  const handleReset = () => {
    setFile(null);
    setJobId(null);
    setStatus(null);
    setResult(null);
    setError(null);
    setClearExisting(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleModeChange = (newMode) => {
    handleReset();
    setMode(newMode);
  };

  const getProgressColor = () => {
    if (status?.status === 'failed') return 'bg-red-500';
    if (status?.status === 'completed') return 'bg-green-500';
    return 'bg-blue-500';
  };

  const getModeConfig = () => {
    switch (mode) {
      case 'stage1':
        return {
          title: 'Stage 1: Trích xuất cấu trúc (VLM)',
          description: 'Upload PDF → VLM trích xuất → JSON Structure',
          fileType: '.pdf',
          fileLabel: 'PDF',
          icon: (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          ),
          color: 'from-purple-500 to-indigo-500',
          showNeo4j: false
        };
      case 'stage2':
        return {
          title: 'Stage 2: Trích xuất ngữ nghĩa (LLM)',
          description: 'Upload Stage 1 JSON → LLM trích xuất → Merged Result',
          fileType: '.json',
          fileLabel: 'JSON',
          icon: (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
          ),
          color: 'from-blue-500 to-cyan-500',
          showNeo4j: true
        };
      case 'neo4j':
        return {
          title: 'Import Neo4j',
          description: 'Upload JSON extraction → Push lên Neo4j database',
          fileType: '.json',
          fileLabel: 'JSON',
          icon: (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
            </svg>
          ),
          color: 'from-orange-500 to-red-500',
          showNeo4j: false,
          showClearOption: true
        };
      case 'weaviate':
        return {
          title: 'Index Weaviate (Vector)',
          description: 'Upload JSON → Embedding → Weaviate vector DB',
          fileType: '.json',
          fileLabel: 'JSON',
          icon: (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          ),
          color: 'from-emerald-500 to-teal-500',
          showNeo4j: false,
          showClearOption: false,
          showDocType: true
        };
      case 'opensearch':
        return {
          title: 'Index OpenSearch (BM25)',
          description: 'Upload JSON → Keyword index → OpenSearch',
          fileType: '.json',
          fileLabel: 'JSON',
          icon: (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          ),
          color: 'from-amber-500 to-orange-500',
          showNeo4j: false,
          showClearOption: true,
          showDocType: true
        };
      default:
        return {
          title: 'Full Pipeline',
          description: 'Upload PDF → VLM + LLM → Complete Result',
          fileType: '.pdf',
          fileLabel: 'PDF',
          icon: (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          ),
          color: 'from-green-500 to-teal-500',
          showNeo4j: true
        };
    }
  };

  const config = getModeConfig();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-3xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b dark:border-gray-700">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 bg-gradient-to-br ${config.color} rounded-lg flex items-center justify-center text-white`}>
              {config.icon}
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Trích xuất Knowledge Graph
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Two-Stage Pipeline: VLM + LLM
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Mode Tabs */}
        <div className="flex border-b dark:border-gray-700">
          <button
            onClick={() => handleModeChange('stage1')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              mode === 'stage1'
                ? 'text-purple-600 border-b-2 border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <span className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/50 text-purple-600 dark:text-purple-400 text-xs flex items-center justify-center font-bold">1</span>
              Stage 1: VLM
            </div>
          </button>
          <button
            onClick={() => handleModeChange('stage2')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              mode === 'stage2'
                ? 'text-blue-600 border-b-2 border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <span className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400 text-xs flex items-center justify-center font-bold">2</span>
              Stage 2: LLM
            </div>
          </button>
          <button
            onClick={() => handleModeChange('full')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              mode === 'full'
                ? 'text-green-600 border-b-2 border-green-500 bg-green-50 dark:bg-green-900/20'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Full
            </div>
          </button>
          <button
            onClick={() => handleModeChange('neo4j')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              mode === 'neo4j'
                ? 'text-orange-600 border-b-2 border-orange-500 bg-orange-50 dark:bg-orange-900/20'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
              Neo4j
            </div>
          </button>
          <button
            onClick={() => handleModeChange('weaviate')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              mode === 'weaviate'
                ? 'text-emerald-600 border-b-2 border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
              Vector
            </div>
          </button>
          <button
            onClick={() => handleModeChange('opensearch')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              mode === 'opensearch'
                ? 'text-amber-600 border-b-2 border-amber-500 bg-amber-50 dark:bg-amber-900/20'
                : 'text-gray-500 hover:text-gray-700 dark:text-gray-400'
            }`}
          >
            <div className="flex items-center justify-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              BM25
            </div>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {/* Mode Description */}
          <div className={`mb-4 p-3 rounded-lg bg-gradient-to-r ${config.color} bg-opacity-10`}>
            <h3 className="font-medium text-gray-900 dark:text-white">{config.title}</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300">{config.description}</p>
          </div>

          {/* Upload Section */}
          {!status && (
            <div className="space-y-4">
              {/* File Drop Zone */}
              <div
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer
                  ${file ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'}`}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={config.fileType}
                  onChange={handleFileChange}
                  className="hidden"
                />
                
                {file ? (
                  <div className="flex items-center justify-center gap-3">
                    <svg className="w-10 h-10 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <div className="text-left">
                      <p className="font-medium text-gray-900 dark:text-white">{file.name}</p>
                      <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  </div>
                ) : (
                  <>
                    <svg className="w-12 h-12 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="text-gray-600 dark:text-gray-300 mb-2">
                      Kéo thả file {config.fileLabel} hoặc click để chọn
                    </p>
                    <p className="text-sm text-gray-400">
                      {mode === 'stage2' 
                        ? 'Upload file JSON từ Stage 1' 
                        : mode === 'neo4j'
                        ? 'Upload file JSON extraction để import vào Neo4j'
                        : mode === 'weaviate'
                        ? 'Upload file JSON extraction để index vào Weaviate (vector search)'
                        : mode === 'opensearch'
                        ? 'Upload file JSON extraction để index vào OpenSearch (keyword search)'
                        : 'Hỗ trợ các văn bản quy định, quy chế'}
                    </p>
                  </>
                )}
              </div>

              {/* Options */}
              <div className="grid grid-cols-2 gap-4">
                {!['neo4j', 'weaviate', 'opensearch'].includes(mode) && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Danh mục
                    </label>
                    <input
                      type="text"
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Quy chế Đào tạo"
                    />
                  </div>
                )}
                {['weaviate', 'opensearch'].includes(mode) && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Loại văn bản
                    </label>
                    <select
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="regulation">Quy chế/Quy định</option>
                      <option value="policy">Chính sách</option>
                      <option value="guide">Hướng dẫn</option>
                      <option value="form">Biểu mẫu</option>
                    </select>
                  </div>
                )}
                {config.showNeo4j && (
                  <div className="flex items-end">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={pushToNeo4j}
                        onChange={(e) => setPushToNeo4j(e.target.checked)}
                        className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        Đẩy lên Neo4j
                      </span>
                    </label>
                  </div>
                )}
                {mode === 'neo4j' && (
                  <div className="col-span-2 flex items-center gap-4">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={clearExisting}
                        onChange={(e) => setClearExisting(e.target.checked)}
                        className="w-4 h-4 text-red-600 rounded focus:ring-red-500"
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        Xóa dữ liệu cũ trước khi import
                      </span>
                    </label>
                    {clearExisting && (
                      <span className="text-xs text-red-500 font-medium">
                        ⚠️ Sẽ xóa toàn bộ dữ liệu hiện tại!
                      </span>
                    )}
                  </div>
                )}
                {mode === 'opensearch' && (
                  <div className="flex items-end">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={clearExisting}
                        onChange={(e) => setClearExisting(e.target.checked)}
                        className="w-4 h-4 text-amber-600 rounded focus:ring-amber-500"
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        Xóa documents cùng file trước khi index
                      </span>
                    </label>
                  </div>
                )}
              </div>

              {/* Error */}
              {error && (
                <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400 text-sm">
                  {error}
                </div>
              )}

              {/* Submit Button */}
              <button
                onClick={handleUpload}
                disabled={!file || isUploading}
                className={`w-full py-3 px-4 rounded-lg font-medium transition-all
                  ${file && !isUploading
                    ? `bg-gradient-to-r ${config.color} text-white hover:opacity-90 shadow-lg hover:shadow-xl`
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed'
                  }`}
              >
                {isUploading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Đang tải lên...
                  </span>
                ) : (
                  `Bắt đầu ${config.title}`
                )}
              </button>
            </div>
          )}

          {/* Processing Status */}
          {status && (
            <div className="space-y-4">
              {/* Progress Bar */}
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-600 dark:text-gray-300">{status.current_step}</span>
                  <span className="font-medium text-gray-900 dark:text-white">{status.progress}%</span>
                </div>
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${getProgressColor()}`}
                    style={{ width: `${status.progress}%` }}
                  />
                </div>
              </div>

              {/* Status Badge */}
              <div className="flex items-center gap-2">
                <span className={`px-3 py-1 rounded-full text-sm font-medium
                  ${status.status === 'completed' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                    status.status === 'failed' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                    'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                  }`}>
                  {status.status === 'completed' ? '✓ Hoàn thành' :
                   status.status === 'failed' ? '✗ Thất bại' :
                   '⟳ Đang xử lý'}
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {status.stage === 'stage1' ? 'Stage 1' : status.stage === 'stage2' ? 'Stage 2' : 'Full Pipeline'}
                </span>
              </div>

              {/* Error Message */}
              {status.error && (
                <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400 text-sm">
                  {status.error}
                </div>
              )}

              {/* Stats */}
              {status.stats && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {(mode === 'stage1' || mode === 'full') && (
                    <>
                      <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                          {status.stats.pages || 0}
                        </div>
                        <div className="text-xs text-gray-500">Trang</div>
                      </div>
                      <div className="p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                          {status.stats.chapters || 0}
                        </div>
                        <div className="text-xs text-gray-500">Chương</div>
                      </div>
                      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                          {status.stats.articles || status.stats.articles_processed || 0}
                        </div>
                        <div className="text-xs text-gray-500">Điều</div>
                      </div>
                      <div className="p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-cyan-600 dark:text-cyan-400">
                          {status.stats.clauses || 0}
                        </div>
                        <div className="text-xs text-gray-500">Khoản</div>
                      </div>
                    </>
                  )}
                  
                  {(mode === 'stage2' || mode === 'full') && status.stats.entities !== undefined && (
                    <>
                      <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {status.stats.entities || 0}
                        </div>
                        <div className="text-xs text-gray-500">Thực thể</div>
                      </div>
                      <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                          {status.stats.relations || 0}
                        </div>
                        <div className="text-xs text-gray-500">Quan hệ</div>
                      </div>
                    </>
                  )}

                  {mode === 'neo4j' && (
                    <>
                      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                          {status.stats?.imported?.documents || 0}
                        </div>
                        <div className="text-xs text-gray-500">Documents</div>
                      </div>
                      <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                          {status.stats?.imported?.articles || 0}
                        </div>
                        <div className="text-xs text-gray-500">Articles</div>
                      </div>
                      <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {status.stats?.imported?.entities || 0}
                        </div>
                        <div className="text-xs text-gray-500">Entities</div>
                      </div>
                      <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                          {(status.stats?.imported?.structural_relations || 0) + (status.stats?.imported?.semantic_relations || 0)}
                        </div>
                        <div className="text-xs text-gray-500">Relations</div>
                      </div>
                    </>
                  )}

                  {(mode === 'weaviate' || mode === 'opensearch') && (
                    <>
                      <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                          {status.stats?.chunks_processed || 0}
                        </div>
                        <div className="text-xs text-gray-500">Chunks xử lý</div>
                      </div>
                      <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg text-center">
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {status.stats?.documents_indexed || 0}
                        </div>
                        <div className="text-xs text-gray-500">Đã index</div>
                      </div>
                      {status.stats?.documents_failed > 0 && (
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-center">
                          <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                            {status.stats?.documents_failed || 0}
                          </div>
                          <div className="text-xs text-gray-500">Lỗi</div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}

              {/* Result Preview */}
              {result && (
                <div className="space-y-3">
                  <h3 className="font-medium text-gray-900 dark:text-white">
                    {['neo4j', 'weaviate', 'opensearch'].includes(mode) ? 'Kết quả Index' : 'Kết quả trích xuất'}
                  </h3>
                  
                  {/* Neo4j Import Success */}
                  {mode === 'neo4j' && (
                    <div className="space-y-3">
                      <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                        <div className="flex items-center gap-2 text-green-700 dark:text-green-300">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span className="font-medium">Import Neo4j thành công!</span>
                        </div>
                        {status.stats?.source_file && (
                          <p className="mt-2 text-sm text-green-600 dark:text-green-400">
                            File: {status.stats.source_file} | Category: {status.stats.category}
                          </p>
                        )}
                      </div>
                      
                      {/* Neo4j Database Stats */}
                      {status.stats?.neo4j_stats && (
                        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                          <h4 className="text-sm font-medium text-blue-700 dark:text-blue-300 mb-2">
                            📊 Tổng dữ liệu trong Neo4j
                          </h4>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            <div className="text-gray-600 dark:text-gray-400">
                              Total Nodes: <span className="font-medium text-blue-600">{status.stats.neo4j_stats.total_nodes || 0}</span>
                            </div>
                            <div className="text-gray-600 dark:text-gray-400">
                              Total Relations: <span className="font-medium text-blue-600">{status.stats.neo4j_stats.total_relationships || 0}</span>
                            </div>
                            {status.stats.neo4j_stats.node_labels && (
                              <div className="col-span-2 text-gray-600 dark:text-gray-400">
                                Labels: {Object.entries(status.stats.neo4j_stats.node_labels).map(([k, v]) => `${k}(${v})`).join(', ')}
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Weaviate/OpenSearch Index Success */}
                  {(mode === 'weaviate' || mode === 'opensearch') && (
                    <div className="space-y-3">
                      <div className={`p-4 border rounded-lg ${
                        mode === 'weaviate' 
                          ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800' 
                          : 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800'
                      }`}>
                        <div className={`flex items-center gap-2 ${
                          mode === 'weaviate' ? 'text-emerald-700 dark:text-emerald-300' : 'text-amber-700 dark:text-amber-300'
                        }`}>
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span className="font-medium">
                            Index {mode === 'weaviate' ? 'Weaviate' : 'OpenSearch'} thành công!
                          </span>
                        </div>
                        {status.stats?.source_file && (
                          <p className={`mt-2 text-sm ${
                            mode === 'weaviate' ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'
                          }`}>
                            File: {status.stats.source_file} | Doc Type: {status.stats.doc_type}
                          </p>
                        )}
                      </div>
                      
                      {/* Index Details */}
                      <div className={`p-4 border rounded-lg ${
                        mode === 'weaviate' 
                          ? 'bg-teal-50 dark:bg-teal-900/20 border-teal-200 dark:border-teal-800' 
                          : 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800'
                      }`}>
                        <h4 className={`text-sm font-medium mb-2 ${
                          mode === 'weaviate' ? 'text-teal-700 dark:text-teal-300' : 'text-orange-700 dark:text-orange-300'
                        }`}>
                          📊 Chi tiết Index
                        </h4>
                        <div className="grid grid-cols-2 gap-2 text-sm text-gray-600 dark:text-gray-400">
                          <div>Chunks: <span className="font-medium">{status.stats?.chunks_processed || 0}</span></div>
                          <div>Indexed: <span className="font-medium text-green-600">{status.stats?.documents_indexed || 0}</span></div>
                          {mode === 'weaviate' && status.stats?.embedding_model && (
                            <div className="col-span-2">Model: <span className="font-medium">{status.stats.embedding_model}</span></div>
                          )}
                          {mode === 'opensearch' && status.stats?.index_name && (
                            <div className="col-span-2">Index: <span className="font-medium">{status.stats.index_name}</span></div>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* Stage 1 Preview */}
                  {(mode === 'stage1' || mode === 'full') && result.structure && (
                    <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-2">
                        Cấu trúc văn bản ({result.structure?.articles?.length || 0} điều)
                      </h4>
                      <div className="max-h-40 overflow-y-auto space-y-1">
                        {result.structure?.articles?.slice(0, 10).map((article, idx) => (
                          <div key={idx} className="text-sm text-gray-700 dark:text-gray-300 p-2 bg-white dark:bg-gray-800 rounded">
                            <span className="font-medium">{article.id}:</span> {article.title?.slice(0, 50)}...
                          </div>
                        ))}
                        {(result.structure?.articles?.length || 0) > 10 && (
                          <div className="text-xs text-gray-500 p-2">
                            +{result.structure.articles.length - 10} điều khác
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Stage 2 Preview */}
                  {(mode === 'stage2' || mode === 'full') && result.stage2_semantic && (
                    <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-2">
                        Thực thể ({result.stage2_semantic?.entities?.length || 0})
                      </h4>
                      <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
                        {result.stage2_semantic?.entities?.slice(0, 20).map((entity, idx) => (
                          <span
                            key={idx}
                            className={`px-2 py-1 rounded-full text-xs font-medium
                              ${entity.type === 'MON_HOC' ? 'bg-pink-100 text-pink-700' :
                                entity.type === 'CHUNG_CHI' ? 'bg-blue-100 text-blue-700' :
                                entity.type === 'QUY_DINH' ? 'bg-purple-100 text-purple-700' :
                                entity.type === 'DIEM_SO' ? 'bg-yellow-100 text-yellow-700' :
                                entity.type === 'DOI_TUONG' ? 'bg-green-100 text-green-700' :
                                'bg-gray-100 text-gray-700'
                              }`}
                          >
                            {entity.text || entity.normalized}
                          </span>
                        ))}
                        {(result.stage2_semantic?.entities?.length || 0) > 20 && (
                          <span className="px-2 py-1 text-xs text-gray-500">
                            +{result.stage2_semantic.entities.length - 20} khác
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Download Buttons */}
                  <div className="flex gap-3">
                    {mode !== 'neo4j' && (
                      <button
                        onClick={handleDownload}
                        className="flex-1 py-2 px-4 bg-green-500 hover:bg-green-600 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Tải JSON {mode === 'stage1' ? 'Stage 1' : mode === 'stage2' ? 'Merged' : ''}
                      </button>
                    )}
                    <button
                      onClick={handleReset}
                      className={`py-2 px-4 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg font-medium hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${mode === 'neo4j' ? 'flex-1' : ''}`}
                    >
                      {mode === 'neo4j' ? 'Import mới' : 'Trích xuất mới'}
                    </button>
                  </div>

                  {/* Next Step Hint for Stage 1 */}
                  {mode === 'stage1' && status.status === 'completed' && (
                    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        💡 Tiếp theo: Tải file JSON này và upload vào <strong>Stage 2</strong> để trích xuất ngữ nghĩa
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default KGExtractionPanel;
