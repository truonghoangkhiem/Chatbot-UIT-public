import React, { useState, useEffect } from 'react';
import { X, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react';
import { getAgentsInfo, checkHealth } from '../services/api';

const SystemInfoModal = ({ isOpen, onClose }) => {
  const [agentsInfo, setAgentsInfo] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isOpen) {
      fetchSystemInfo();
    }
  }, [isOpen]);

  const fetchSystemInfo = async () => {
    setLoading(true);
    try {
      const [agents, health] = await Promise.all([
        getAgentsInfo(),
        checkHealth()
      ]);
      setAgentsInfo(agents);
      setHealthStatus(health);
    } catch (error) {
      console.error('Error fetching system info:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-lg bg-white shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-200 p-4 sticky top-0 bg-white">
          <h2 className="text-xl font-semibold">Thông tin hệ thống</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchSystemInfo}
              className="text-gray-400 hover:text-gray-600"
              disabled={loading}
            >
              <RefreshCw className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
            </div>
          ) : (
            <>
              {/* Health Status */}
              {healthStatus && (
                <div className="rounded-lg border border-gray-200 p-4">
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    {healthStatus.status === 'healthy' ? (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-red-600" />
                    )}
                    Trạng thái hệ thống
                  </h3>
                  <div className="space-y-2">
                    {Object.entries(healthStatus.services || {}).map(([service, status]) => (
                      <div key={service} className="flex items-center justify-between text-sm">
                        <span className="text-gray-700 capitalize">{service}:</span>
                        <span className={`font-medium ${
                          status === 'healthy' || status === 'ok' 
                            ? 'text-green-600' 
                            : 'text-red-600'
                        }`}>
                          {status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Agents Info */}
              {agentsInfo && agentsInfo.multi_agent_system && (
                <div className="rounded-lg border border-gray-200 p-4">
                  <h3 className="font-semibold mb-3">Multi-Agent System</h3>
                  
                  {/* Pipeline Steps */}
                  {agentsInfo.multi_agent_system.pipeline_steps && (
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Pipeline:</h4>
                      <ol className="space-y-1 text-sm text-gray-600">
                        {agentsInfo.multi_agent_system.pipeline_steps.map((step, index) => (
                          <li key={index}>{step}</li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {/* Models Used */}
                  {agentsInfo.models_used && (
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Models:</h4>
                      <div className="space-y-1 text-sm">
                        {Object.entries(agentsInfo.models_used).map(([agent, model]) => (
                          <div key={agent} className="flex justify-between">
                            <span className="text-gray-600 capitalize">{agent}:</span>
                            <span className="text-gray-900 font-mono text-xs">{model}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Capabilities */}
                  {agentsInfo.capabilities && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Khả năng:</h4>
                      <div className="space-y-2">
                        {Object.entries(agentsInfo.capabilities).map(([capability, description]) => (
                          <div key={capability} className="text-sm">
                            <span className="font-medium text-gray-700 capitalize">
                              {capability.replace(/_/g, ' ')}:
                            </span>
                            <p className="text-gray-600 ml-4">{description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 p-4 bg-gray-50">
          <p className="text-xs text-gray-500 text-center">
            Chatbot UIT - Multi-Agent RAG System v1.0.0
          </p>
        </div>
      </div>
    </div>
  );
};

export default SystemInfoModal;
