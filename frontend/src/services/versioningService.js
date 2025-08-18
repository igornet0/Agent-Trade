import api from './api';

/**
 * Сервис для работы с версионированием моделей
 */
const versioningService = {
  /**
   * Создание новой версии модели
   */
  async createVersion(agentId, version, modelPath, configPath = null, scalerPath = null, metadata = null) {
    try {
      const response = await api.post(`/api_db_agent/models/${agentId}/versions`, {
        version,
        model_path: modelPath,
        config_path: configPath,
        scaler_path: scalerPath,
        metadata
      });
      return response.data;
    } catch (error) {
      console.error('Error creating model version:', error);
      throw error;
    }
  },

  /**
   * Продвижение версии в продакшн
   */
  async promoteVersion(agentId, version, force = false) {
    try {
      const response = await api.post(`/api_db_agent/models/${agentId}/versions/${version}/promote`, {
        force
      });
      return response.data;
    } catch (error) {
      console.error('Error promoting model version:', error);
      throw error;
    }
  },

  /**
   * Откат к указанной версии
   */
  async rollbackVersion(agentId, version) {
    try {
      const response = await api.post(`/api_db_agent/models/${agentId}/versions/${version}/rollback`);
      return response.data;
    } catch (error) {
      console.error('Error rolling back model version:', error);
      throw error;
    }
  },

  /**
   * Получение списка версий модели
   */
  async listVersions(agentId, limit = null) {
    try {
      const params = new URLSearchParams();
      if (limit) params.append('limit', limit);

      const response = await api.get(`/api_db_agent/models/${agentId}/versions?${params}`);
      return response.data;
    } catch (error) {
      console.error('Error listing model versions:', error);
      throw error;
    }
  },

  /**
   * Получение детальной информации о версии
   */
  async getVersionInfo(agentId, version) {
    try {
      const response = await api.get(`/api_db_agent/models/${agentId}/versions/${version}`);
      return response.data;
    } catch (error) {
      console.error('Error getting model version info:', error);
      throw error;
    }
  },

  /**
   * Удаление версии модели
   */
  async deleteVersion(agentId, version, force = false) {
    try {
      const params = new URLSearchParams();
      if (force) params.append('force', 'true');

      const response = await api.delete(`/api_db_agent/models/${agentId}/versions/${version}?${params}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting model version:', error);
      throw error;
    }
  },

  /**
   * Получение статуса продакшн версии
   */
  async getProductionStatus(agentId) {
    try {
      const response = await api.get(`/api_db_agent/models/${agentId}/production`);
      return response.data;
    } catch (error) {
      console.error('Error getting model production status:', error);
      throw error;
    }
  },

  /**
   * Очистка старых версий
   */
  async cleanupVersions(agentId, keepVersions = 5) {
    try {
      const response = await api.post(`/api_db_agent/models/${agentId}/versions/cleanup`, {
        keep_versions: keepVersions
      });
      return response.data;
    } catch (error) {
      console.error('Error cleaning up model versions:', error);
      throw error;
    }
  },

  /**
   * Форматирование размера файла
   */
  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  /**
   * Форматирование даты создания
   */
  formatCreatedAt(createdAt) {
    if (!createdAt) return 'Unknown';
    
    try {
      const date = new Date(createdAt);
      return date.toLocaleString();
    } catch (error) {
      return 'Invalid date';
    }
  },

  /**
   * Получение статуса версии для отображения
   */
  getVersionStatus(versionInfo) {
    if (versionInfo.is_production) {
      return {
        label: 'Production',
        color: 'success',
        icon: 'check-circle'
      };
    } else {
      return {
        label: 'Development',
        color: 'secondary',
        icon: 'code'
      };
    }
  },

  /**
   * Проверка целостности версии
   */
  checkVersionIntegrity(versionInfo) {
    const requiredArtifacts = ['model'];
    const existingArtifacts = versionInfo.artifacts.map(a => a.type);
    
    const missing = requiredArtifacts.filter(type => !existingArtifacts.includes(type));
    const corrupted = versionInfo.artifacts.filter(a => !a.exists);
    
    return {
      isValid: missing.length === 0 && corrupted.length === 0,
      missing,
      corrupted: corrupted.map(a => a.type),
      totalSize: versionInfo.total_size || 0
    };
  },

  /**
   * Сравнение версий
   */
  compareVersions(version1, version2) {
    const comparison = {
      version1: {
        version: version1.version,
        totalSize: version1.total_size || 0,
        artifactCount: version1.artifacts?.length || 0,
        isProduction: version1.is_production || false
      },
      version2: {
        version: version2.version,
        totalSize: version2.total_size || 0,
        artifactCount: version2.artifacts?.length || 0,
        isProduction: version2.is_production || false
      },
      differences: []
    };

    // Сравниваем размеры
    if (version1.total_size !== version2.total_size) {
      comparison.differences.push({
        type: 'size',
        field: 'total_size',
        value1: version1.total_size,
        value2: version2.total_size
      });
    }

    // Сравниваем количество артефактов
    const artifacts1 = version1.artifacts?.length || 0;
    const artifacts2 = version2.artifacts?.length || 0;
    if (artifacts1 !== artifacts2) {
      comparison.differences.push({
        type: 'artifacts',
        field: 'artifact_count',
        value1: artifacts1,
        value2: artifacts2
      });
    }

    // Сравниваем типы артефактов
    const types1 = new Set(version1.artifacts?.map(a => a.type) || []);
    const types2 = new Set(version2.artifacts?.map(a => a.type) || []);
    
    const missingIn1 = [...types2].filter(type => !types1.has(type));
    const missingIn2 = [...types1].filter(type => !types2.has(type));
    
    if (missingIn1.length > 0 || missingIn2.length > 0) {
      comparison.differences.push({
        type: 'artifacts',
        field: 'artifact_types',
        value1: [...types1],
        value2: [...types2],
        missingIn1,
        missingIn2
      });
    }

    return comparison;
  },

  /**
   * Получение рекомендаций по версионированию
   */
  getVersioningRecommendations(versions) {
    const recommendations = [];

    if (versions.length === 0) {
      recommendations.push({
        type: 'info',
        message: 'No versions found. Create your first version.',
        priority: 'high'
      });
      return recommendations;
    }

    // Проверяем наличие продакшн версии
    const hasProduction = versions.some(v => v.is_production);
    if (!hasProduction) {
      recommendations.push({
        type: 'warning',
        message: 'No production version set. Consider promoting a stable version.',
        priority: 'high'
      });
    }

    // Проверяем количество версий
    if (versions.length > 10) {
      recommendations.push({
        type: 'info',
        message: 'Many versions found. Consider cleaning up old versions.',
        priority: 'medium'
      });
    }

    // Проверяем размеры версий
    const largeVersions = versions.filter(v => (v.total_size || 0) > 100 * 1024 * 1024); // > 100MB
    if (largeVersions.length > 0) {
      recommendations.push({
        type: 'warning',
        message: `${largeVersions.length} large version(s) found. Consider optimizing model size.`,
        priority: 'medium'
      });
    }

    return recommendations;
  }
};

export default versioningService;
