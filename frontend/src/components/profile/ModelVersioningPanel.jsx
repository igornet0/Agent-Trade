import React, { useState, useEffect } from 'react';
import versioningService from '../../services/versioningService';

const ModelVersioningPanel = ({ agentId, onVersionChange }) => {
  const [versions, setVersions] = useState([]);
  const [productionStatus, setProductionStatus] = useState(null);
  const [selectedVersion, setSelectedVersion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showPromoteModal, setShowPromoteModal] = useState(false);
  const [showRollbackModal, setShowRollbackModal] = useState(false);

  // Форма для создания версии
  const [newVersion, setNewVersion] = useState({
    version: '',
    modelPath: '',
    configPath: '',
    scalerPath: '',
    metadata: {}
  });

  useEffect(() => {
    if (agentId) {
      loadVersions();
      loadProductionStatus();
    }
  }, [agentId]);

  const loadVersions = async () => {
    try {
      setLoading(true);
      const versionsData = await versioningService.listVersions(agentId);
      setVersions(versionsData);
    } catch (err) {
      setError('Failed to load versions: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadProductionStatus = async () => {
    try {
      const status = await versioningService.getProductionStatus(agentId);
      setProductionStatus(status);
    } catch (err) {
      console.error('Failed to load production status:', err);
    }
  };

  const handleCreateVersion = async () => {
    try {
      setLoading(true);
      await versioningService.createVersion(
        agentId,
        newVersion.version,
        newVersion.modelPath,
        newVersion.configPath || null,
        newVersion.scalerPath || null,
        Object.keys(newVersion.metadata).length > 0 ? newVersion.metadata : null
      );
      
      setShowCreateModal(false);
      setNewVersion({ version: '', modelPath: '', configPath: '', scalerPath: '', metadata: {} });
      await loadVersions();
      
      if (onVersionChange) {
        onVersionChange();
      }
    } catch (err) {
      setError('Failed to create version: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handlePromoteVersion = async (version) => {
    try {
      setLoading(true);
      await versioningService.promoteVersion(agentId, version, false);
      setShowPromoteModal(false);
      await loadVersions();
      await loadProductionStatus();
      
      if (onVersionChange) {
        onVersionChange();
      }
    } catch (err) {
      setError('Failed to promote version: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRollbackVersion = async (version) => {
    try {
      setLoading(true);
      await versioningService.rollbackVersion(agentId, version);
      setShowRollbackModal(false);
      await loadVersions();
      await loadProductionStatus();
      
      if (onVersionChange) {
        onVersionChange();
      }
    } catch (err) {
      setError('Failed to rollback version: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteVersion = async (version) => {
    if (!window.confirm(`Are you sure you want to delete version ${version}?`)) {
      return;
    }

    try {
      setLoading(true);
      await versioningService.deleteVersion(agentId, version, false);
      await loadVersions();
      
      if (onVersionChange) {
        onVersionChange();
      }
    } catch (err) {
      setError('Failed to delete version: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCleanupVersions = async () => {
    if (!window.confirm('Are you sure you want to cleanup old versions? This will keep only the 5 most recent versions.')) {
      return;
    }

    try {
      setLoading(true);
      await versioningService.cleanupVersions(agentId, 5);
      await loadVersions();
    } catch (err) {
      setError('Failed to cleanup versions: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const getVersionStatus = (versionInfo) => {
    return versioningService.getVersionStatus(versionInfo);
  };

  const formatFileSize = (bytes) => {
    return versioningService.formatFileSize(bytes);
  };

  if (loading && versions.length === 0) {
    return (
      <div className="flex justify-center items-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Model Versions</h3>
        <div className="space-x-2">
          <button
            onClick={() => setShowCreateModal(true)}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
          >
            Create Version
          </button>
          <button
            onClick={handleCleanupVersions}
            className="bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 transition-colors"
          >
            Cleanup
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {/* Production Status */}
      {productionStatus && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <h4 className="font-medium text-green-800 mb-2">Production Status</h4>
          {productionStatus.has_production ? (
            <div className="text-green-700">
              <p>Current Production Version: <span className="font-semibold">{productionStatus.production_version}</span></p>
              {productionStatus.version_info && (
                <p className="text-sm mt-1">
                  Size: {formatFileSize(productionStatus.version_info.total_size)} | 
                  Artifacts: {productionStatus.version_info.artifacts?.length || 0}
                </p>
              )}
            </div>
          ) : (
            <p className="text-yellow-700">No production version set</p>
          )}
        </div>
      )}

      {/* Versions List */}
      <div className="space-y-4">
        {versions.map((versionInfo) => {
          const status = getVersionStatus(versionInfo);
          const integrity = versioningService.checkVersionIntegrity(versionInfo);
          
          return (
            <div
              key={versionInfo.version}
              className={`border rounded-lg p-4 ${
                versionInfo.is_production ? 'border-green-300 bg-green-50' : 'border-gray-200'
              }`}
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="flex items-center space-x-3">
                    <h4 className="text-lg font-medium text-gray-900">
                      Version {versionInfo.version}
                    </h4>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      status.color === 'success' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {status.label}
                    </span>
                    {!integrity.isValid && (
                      <span className="px-2 py-1 text-xs font-medium rounded-full bg-red-100 text-red-800">
                        Corrupted
                      </span>
                    )}
                  </div>
                  
                  <div className="mt-2 text-sm text-gray-600">
                    <p>Size: {formatFileSize(versionInfo.total_size)}</p>
                    <p>Artifacts: {versionInfo.artifacts?.length || 0}</p>
                    {versionInfo.artifacts && (
                      <p>Types: {versionInfo.artifacts.map(a => a.type).join(', ')}</p>
                    )}
                  </div>
                </div>
                
                <div className="flex space-x-2">
                  {!versionInfo.is_production && (
                    <>
                      <button
                        onClick={() => {
                          setSelectedVersion(versionInfo.version);
                          setShowPromoteModal(true);
                        }}
                        className="bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700 transition-colors"
                      >
                        Promote
                      </button>
                      <button
                        onClick={() => {
                          setSelectedVersion(versionInfo.version);
                          setShowRollbackModal(true);
                        }}
                        className="bg-yellow-600 text-white px-3 py-1 rounded text-sm hover:bg-yellow-700 transition-colors"
                      >
                        Rollback
                      </button>
                    </>
                  )}
                  <button
                    onClick={() => handleDeleteVersion(versionInfo.version)}
                    className="bg-red-600 text-white px-3 py-1 rounded text-sm hover:bg-red-700 transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {versions.length === 0 && !loading && (
        <div className="text-center py-8 text-gray-500">
          No versions found. Create your first version to get started.
        </div>
      )}

      {/* Create Version Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Create New Version</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Version
                </label>
                <input
                  type="text"
                  value={newVersion.version}
                  onChange={(e) => setNewVersion({...newVersion, version: e.target.value})}
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  placeholder="e.g., 1.0.0"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model Path
                </label>
                <input
                  type="text"
                  value={newVersion.modelPath}
                  onChange={(e) => setNewVersion({...newVersion, modelPath: e.target.value})}
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  placeholder="/path/to/model.pth"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Config Path (optional)
                </label>
                <input
                  type="text"
                  value={newVersion.configPath}
                  onChange={(e) => setNewVersion({...newVersion, configPath: e.target.value})}
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  placeholder="/path/to/config.json"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Scaler Path (optional)
                </label>
                <input
                  type="text"
                  value={newVersion.scalerPath}
                  onChange={(e) => setNewVersion({...newVersion, scalerPath: e.target.value})}
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  placeholder="/path/to/scaler.pkl"
                />
              </div>
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateVersion}
                disabled={!newVersion.version || !newVersion.modelPath || loading}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Promote Version Modal */}
      {showPromoteModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Promote Version</h3>
            <p className="text-gray-600 mb-4">
              Are you sure you want to promote version <strong>{selectedVersion}</strong> to production?
            </p>
            
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowPromoteModal(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={() => handlePromoteVersion(selectedVersion)}
                disabled={loading}
                className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50"
              >
                Promote
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Rollback Version Modal */}
      {showRollbackModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Rollback Version</h3>
            <p className="text-gray-600 mb-4">
              Are you sure you want to rollback to version <strong>{selectedVersion}</strong>?
            </p>
            
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowRollbackModal(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={() => handleRollbackVersion(selectedVersion)}
                disabled={loading}
                className="bg-yellow-600 text-white px-4 py-2 rounded-md hover:bg-yellow-700 disabled:opacity-50"
              >
                Rollback
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelVersioningPanel;
