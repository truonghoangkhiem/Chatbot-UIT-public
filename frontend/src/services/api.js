import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds timeout for LLM responses
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for debugging
apiClient.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.status, error.message);
    return Promise.reject(error);
  }
);

/**
 * Chat API - Send message to chatbot
 * Backend automatically decides RAG usage and parameters via SmartPlannerAgent
 * @param {string} query - User's message
 * @param {string} sessionId - Session ID for conversation
 * @returns {Promise} API response with chatbot reply
 */
export const sendChatMessage = async (query, sessionId) => {
  try {
    const response = await apiClient.post('/chat', {
      query,
      session_id: sessionId,
      // Let backend decide automatically via SmartPlannerAgent
      use_rag: true,  // Always enable, SmartPlanner will decide if actually needed
    });
    return response.data;
  } catch (error) {
    console.error('Error sending chat message:', error);
    throw error;
  }
};

/**
 * Simple Chat API - Faster but less sophisticated responses
 * @param {string} query - User's message
 * @param {string} sessionId - Session ID for conversation
 * @param {Object} options - Additional options
 * @returns {Promise} API response with chatbot reply
 */
export const sendSimpleChatMessage = async (query, sessionId, options = {}) => {
  const {
    useRag = true,
    ragTopK = 5,
    model = null,
    temperature = 0.7,
    maxTokens = 2000,
  } = options;

  try {
    const response = await apiClient.post('/chat/simple', {
      query,
      session_id: sessionId,
      use_rag: useRag,
      rag_top_k: ragTopK,
      model,
      temperature,
      max_tokens: maxTokens,
      stream: false,
    });
    return response.data;
  } catch (error) {
    console.error('Error sending simple chat message:', error);
    throw error;
  }
};

/**
 * Health Check API
 * @returns {Promise} Health status of all services
 */
export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    console.error('Error checking health:', error);
    throw error;
  }
};

/**
 * Get all active conversations
 * @returns {Promise} List of active conversations
 */
export const getConversations = async () => {
  try {
    const response = await apiClient.get('/conversations');
    return response.data;
  } catch (error) {
    console.error('Error fetching conversations:', error);
    throw error;
  }
};

/**
 * Delete a conversation
 * @param {string} sessionId - Session ID to delete
 * @returns {Promise} Deletion confirmation
 */
export const deleteConversation = async (sessionId) => {
  try {
    const response = await apiClient.delete(`/conversations/${sessionId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting conversation:', error);
    throw error;
  }
};

/**
 * Get agent system information
 * @returns {Promise} Information about multi-agent system
 */
export const getAgentsInfo = async () => {
  try {
    const response = await apiClient.get('/agents/info');
    return response.data;
  } catch (error) {
    console.error('Error fetching agents info:', error);
    throw error;
  }
};

/**
 * Test multi-agent system
 * @returns {Promise} Test results
 */
export const testAgents = async () => {
  try {
    const response = await apiClient.post('/agents/test');
    return response.data;
  } catch (error) {
    console.error('Error testing agents:', error);
    throw error;
  }
};

// ============================================================================
// KG EXTRACTION API
// ============================================================================

const RAG_SERVICE_URL = import.meta.env.VITE_RAG_SERVICE_URL || 'http://localhost:8002';

/**
 * Upload PDF and start KG extraction
 * @param {File} file - PDF file to extract
 * @param {string} category - Document category
 * @param {boolean} pushToNeo4j - Whether to push to Neo4j
 * @returns {Promise} Job status
 */
export const uploadForExtraction = async (file, category = 'Quy chế Đào tạo', pushToNeo4j = false) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const url = new URL(`${RAG_SERVICE_URL}/v1/extraction/upload`);
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

  return response.json();
};

/**
 * Get extraction job status
 * @param {string} jobId - Job ID
 * @returns {Promise} Job status
 */
export const getExtractionStatus = async (jobId) => {
  const response = await fetch(`${RAG_SERVICE_URL}/v1/extraction/status/${jobId}`);
  if (!response.ok) {
    throw new Error('Failed to get status');
  }
  return response.json();
};

/**
 * Get extraction result
 * @param {string} jobId - Job ID
 * @returns {Promise} Extraction result JSON
 */
export const getExtractionResult = async (jobId) => {
  const response = await fetch(`${RAG_SERVICE_URL}/v1/extraction/result/${jobId}`);
  if (!response.ok) {
    throw new Error('Failed to get result');
  }
  return response.json();
};

/**
 * Upload JSON file to Neo4j
 * @param {File} file - JSON file with extraction data
 * @param {boolean} clearExisting - Whether to clear existing data before import
 * @returns {Promise} Import result
 */
export const uploadToNeo4j = async (file, clearExisting = false) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const url = new URL(`${RAG_SERVICE_URL}/v1/extraction/neo4j/upload`);
  url.searchParams.append('clear_existing', clearExisting);

  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Neo4j import failed');
  }

  return response.json();
};

/**
 * Get Neo4j database statistics
 * @returns {Promise} Database stats
 */
export const getNeo4jStats = async () => {
  const response = await fetch(`${RAG_SERVICE_URL}/v1/extraction/neo4j/stats`);
  if (!response.ok) {
    throw new Error('Failed to get Neo4j stats');
  }
  return response.json();
};

/**
 * Clear all Neo4j data
 * @returns {Promise} Clear result
 */
export const clearNeo4j = async () => {
  const response = await fetch(`${RAG_SERVICE_URL}/v1/extraction/neo4j/clear`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to clear Neo4j');
  }
  return response.json();
};

export default {
  sendChatMessage,
  sendSimpleChatMessage,
  checkHealth,
  getConversations,
  deleteConversation,
  getAgentsInfo,
  testAgents,
  uploadForExtraction,
  getExtractionStatus,
  getExtractionResult,
  uploadToNeo4j,
  getNeo4jStats,
  clearNeo4j,
};

