import api from './api';

/**
 * Сервис для работы с пайплайнами и бэктестами
 */
const pipelineService = {
  /**
   * Запуск пайплайна
   */
  async runPipeline(pipelineConfig, timeframe, startDate, endDate, coins) {
    try {
      const response = await api.post('/api_db_agent/pipeline/run', {
        pipeline_config: pipelineConfig,
        timeframe,
        start_date: startDate,
        end_date: endDate,
        coins
      });
      return response.data;
    } catch (error) {
      console.error('Error running pipeline:', error);
      throw error;
    }
  },

  /**
   * Получение статуса задачи пайплайна
   */
  async getTaskStatus(taskId) {
    try {
      const response = await api.get(`/api_db_agent/pipeline/tasks/${taskId}`);
      return response.data;
    } catch (error) {
      console.error('Error getting task status:', error);
      throw error;
    }
  },

  /**
   * Отмена задачи пайплайна
   */
  async revokeTask(taskId) {
    try {
      const response = await api.post(`/api_db_agent/pipeline/tasks/${taskId}/revoke`);
      return response.data;
    } catch (error) {
      console.error('Error revoking task:', error);
      throw error;
    }
  },

  /**
   * Получение списка бэктестов
   */
  async listBacktests(pipelineId = null, status = null, limit = 50) {
    try {
      const params = new URLSearchParams();
      if (pipelineId) params.append('pipeline_id', pipelineId);
      if (status) params.append('status', status);
      if (limit) params.append('limit', limit);

      const response = await api.get(`/api_db_agent/pipeline/backtests?${params}`);
      return response.data;
    } catch (error) {
      console.error('Error listing backtests:', error);
      throw error;
    }
  },

  /**
   * Получение детальной информации о бэктесте
   */
  async getBacktest(backtestId) {
    try {
      const response = await api.get(`/api_db_agent/pipeline/backtests/${backtestId}`);
      return response.data;
    } catch (error) {
      console.error('Error getting backtest:', error);
      throw error;
    }
  },

  /**
   * Скачивание артефакта
   */
  async downloadArtifact(artifactPath) {
    try {
      const response = await api.get(`/api_db_agent/pipeline/artifacts/${artifactPath}`, {
        responseType: 'blob'
      });
      
      // Создаем ссылку для скачивания
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', artifactPath.split('/').pop());
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      return true;
    } catch (error) {
      console.error('Error downloading artifact:', error);
      throw error;
    }
  },

  /**
   * Поллинг статуса задачи с автоматическим обновлением
   */
  async pollTaskStatus(taskId, onProgress, onComplete, onError, interval = 2000) {
    const poll = async () => {
      try {
        const status = await this.getTaskStatus(taskId);
        
        if (status.state === 'SUCCESS') {
          onComplete(status);
          return;
        } else if (status.state === 'FAILURE') {
          onError(status);
          return;
        } else if (status.state === 'PROGRESS' || status.state === 'PENDING') {
          onProgress(status);
          // Продолжаем поллинг
          setTimeout(poll, interval);
        }
      } catch (error) {
        onError(error);
      }
    };

    // Начинаем поллинг
    poll();
  },

  /**
   * Форматирование прогресса для отображения
   */
  formatProgress(status) {
    if (status.state === 'PENDING') {
      return {
        percentage: 0,
        status: 'Ожидание...',
        color: 'warning'
      };
    } else if (status.state === 'PROGRESS') {
      const percentage = Math.round((status.current / status.total) * 100);
      return {
        percentage,
        status: status.status || 'Выполняется...',
        color: 'primary'
      };
    } else if (status.state === 'SUCCESS') {
      return {
        percentage: 100,
        status: 'Завершено',
        color: 'success'
      };
    } else {
      return {
        percentage: 0,
        status: status.status || 'Ошибка',
        color: 'danger'
      };
    }
  },

  /**
   * Форматирование метрик для отображения
   */
  formatMetrics(metrics) {
    if (!metrics) return {};

    const formatted = {
      overall: {},
      perCoin: {},
      portfolio: {}
    };

    // Общие метрики
    if (metrics.overall) {
      formatted.overall = {
        totalCoins: metrics.overall.total_coins || 0,
        successfulPredictions: metrics.overall.successful_predictions || 0,
        successfulTrades: metrics.overall.successful_trades || 0,
        averageRiskScore: (metrics.overall.average_risk_score || 0).toFixed(3)
      };
    }

    // Метрики по монетам
    if (metrics.per_coin) {
      formatted.perCoin = Object.entries(metrics.per_coin).reduce((acc, [coin, coinMetrics]) => {
        acc[coin] = {
          predTime: coinMetrics.pred_time || {},
          tradeTime: coinMetrics.trade_time || {},
          risk: coinMetrics.risk || {}
        };
        return acc;
      }, {});
    }

    // Портфельные метрики
    if (metrics.portfolio) {
      formatted.portfolio = {
        totalReturn: (metrics.portfolio.total_return || 0).toFixed(4),
        sharpeRatio: (metrics.portfolio.sharpe_ratio || 0).toFixed(3),
        maxDrawdown: (metrics.portfolio.max_drawdown || 0).toFixed(4),
        winRate: (metrics.portfolio.win_rate || 0).toFixed(3),
        totalTrades: metrics.portfolio.total_trades || 0
      };
    }

    return formatted;
  },

  /**
   * Получение статуса бэктеста в читаемом виде
   */
  getBacktestStatus(status) {
    const statusMap = {
      'running': { label: 'Выполняется', color: 'primary' },
      'completed': { label: 'Завершен', color: 'success' },
      'failed': { label: 'Ошибка', color: 'danger' },
      'cancelled': { label: 'Отменен', color: 'warning' }
    };

    return statusMap[status] || { label: status, color: 'secondary' };
  }
};

export default pipelineService;


