import React, { useState, useEffect } from 'react';
import { getBacktests, getBacktest } from '../../services/pipelineService';

const BacktestHistory = () => {
  const [backtests, setBacktests] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedBacktest, setSelectedBacktest] = useState(null);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    loadBacktests();
  }, []);

  const loadBacktests = async () => {
    try {
      setLoading(true);
      const data = await getBacktests();
      setBacktests(data);
    } catch (error) {
      console.error('Error loading backtests:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleBacktestSelect = async (backtestId) => {
    try {
      const data = await getBacktest(backtestId);
      setSelectedBacktest(data);
      setShowDetails(true);
    } catch (error) {
      console.error('Error loading backtest details:', error);
    }
  };

  const downloadArtifact = async (path, filename) => {
    try {
      const response = await fetch(`/api/pipeline/artifacts/${path}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Error downloading artifact:', error);
    }
  };

  const getStatusColor = (status) => {
    const colorMap = {
      'completed': 'bg-green-100 text-green-800',
      'running': 'bg-blue-100 text-blue-800',
      'failed': 'bg-red-100 text-red-800',
      'pending': 'bg-yellow-100 text-yellow-800'
    };
    return colorMap[status] || 'bg-gray-100 text-gray-800';
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('ru-RU');
  };

  const formatMetrics = (metrics) => {
    if (!metrics || typeof metrics !== 'object') return 'Нет данных';
    
    const keyMetrics = {
      'Sharpe': metrics.Sharpe,
      'PnL': metrics.PnL,
      'Win Rate': metrics.win_rate,
      'Max DD': metrics.max_drawdown,
      'Turnover': metrics.turnover_rate
    };
    
    return Object.entries(keyMetrics)
      .filter(([_, value]) => value !== undefined)
      .map(([key, value]) => `${key}: ${typeof value === 'number' ? value.toFixed(4) : value}`)
      .join(', ');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Загрузка истории бэктестов...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-900">История бэктестов</h2>
        <button
          onClick={loadBacktests}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Обновить
        </button>
      </div>

      {backtests.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-400 text-6xl mb-4">📊</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">Бэктесты не найдены</h3>
          <p className="text-gray-500">Запустите первый бэктест для начала работы</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {backtests.map((backtest) => (
            <div
              key={backtest.id}
              className="bg-white border border-gray-200 rounded-lg shadow hover:shadow-lg transition-shadow cursor-pointer"
              onClick={() => handleBacktestSelect(backtest.id)}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Бэктест #{backtest.id}
                  </h3>
                  <span className={`px-2 py-1 text-xs rounded-full font-medium ${getStatusColor(backtest.status)}`}>
                    {backtest.status}
                  </span>
                </div>
                
                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Таймфрейм:</span>
                    <span className="font-medium">{backtest.timeframe}</span>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Период:</span>
                    <span className="font-medium">
                      {formatDate(backtest.start_date)} - {formatDate(backtest.end_date)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Создан:</span>
                    <span className="font-medium">{formatDate(backtest.created_at)}</span>
                  </div>
                </div>

                {backtest.metrics && (
                  <div className="mb-4">
                    <div className="text-xs text-gray-600 mb-1">Ключевые метрики</div>
                    <div className="text-sm text-gray-800 bg-gray-50 p-2 rounded">
                      {formatMetrics(backtest.metrics)}
                    </div>
                  </div>
                )}

                <div className="flex justify-end">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleBacktestSelect(backtest.id);
                    }}
                    className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                  >
                    Подробнее →
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Backtest Details Modal */}
      {showDetails && selectedBacktest && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-gray-900">
                    Бэктест #{selectedBacktest.id}
                  </h3>
                  <p className="text-gray-600 mt-1">
                    {formatDate(selectedBacktest.created_at)}
                  </p>
                </div>
                <button
                  onClick={() => setShowDetails(false)}
                  className="text-gray-500 hover:text-gray-700 text-2xl"
                >
                  ×
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="space-y-4">
                  <div>
                    <h4 className="text-lg font-semibold text-gray-900 mb-2">Конфигурация</h4>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Таймфрейм:</span>
                          <span className="font-medium">{selectedBacktest.timeframe}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Начало:</span>
                          <span className="font-medium">{formatDate(selectedBacktest.start_date)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Конец:</span>
                          <span className="font-medium">{formatDate(selectedBacktest.end_date)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Статус:</span>
                          <span className={`px-2 py-1 text-xs rounded-full font-medium ${getStatusColor(selectedBacktest.status)}`}>
                            {selectedBacktest.status}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {selectedBacktest.config_json && (
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 mb-2">Конфигурация пайплайна</h4>
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <pre className="text-xs text-gray-700 overflow-auto max-h-32">
                          {JSON.stringify(selectedBacktest.config_json, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <div>
                    <h4 className="text-lg font-semibold text-gray-900 mb-2">Метрики</h4>
                    {selectedBacktest.metrics ? (
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="grid grid-cols-2 gap-3">
                          {Object.entries(selectedBacktest.metrics).map(([key, value]) => (
                            <div key={key} className="text-sm">
                              <span className="text-gray-600">{key}:</span>
                              <span className="ml-2 font-medium">
                                {typeof value === 'number' ? value.toFixed(4) : String(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="bg-gray-50 p-4 rounded-lg text-gray-500 text-sm">
                        Метрики недоступны
                      </div>
                    )}
                  </div>

                  <div>
                    <h4 className="text-lg font-semibold text-gray-900 mb-2">Артефакты</h4>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <div className="space-y-2">
                        <button
                          onClick={() => downloadArtifact(selectedBacktest.artifact_path + '/equity.csv', 'equity.csv')}
                          className="w-full text-left px-3 py-2 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded text-sm transition-colors"
                        >
                          📊 Скачать equity curve (CSV)
                        </button>
                        <button
                          onClick={() => downloadArtifact(selectedBacktest.artifact_path + '/trades.csv', 'trades.csv')}
                          className="w-full text-left px-3 py-2 bg-green-100 hover:bg-green-200 text-green-800 rounded text-sm transition-colors"
                        >
                          💰 Скачать trades (CSV)
                        </button>
                        {selectedBacktest.artifact_path && (
                          <div className="text-xs text-gray-500 mt-2">
                            Путь: {selectedBacktest.artifact_path}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
                <button
                  onClick={() => setShowDetails(false)}
                  className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  Закрыть
                </button>
                <button
                  onClick={() => {
                    // TODO: Implement backtest rerun
                    console.log('Rerun backtest:', selectedBacktest.id);
                  }}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                >
                  Запустить заново
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BacktestHistory;
