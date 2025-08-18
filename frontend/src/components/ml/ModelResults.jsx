import React, { useState, useEffect } from 'react';
import { getModelMetrics } from '../../services/mlService';

export default function ModelResults({ modelId, results }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (modelId) {
      loadMetrics();
    }
  }, [modelId]);

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const data = await getModelMetrics(modelId);
      setMetrics(data);
    } catch (e) {
      console.error('Ошибка загрузки метрик:', e);
    } finally {
      setLoading(false);
    }
  };

  const formatMetric = (metric, value) => {
    if (typeof value === 'number') {
      if (metric.includes('ratio') || metric.includes('rate') || metric.includes('accuracy')) {
        return `${(value * 100).toFixed(2)}%`;
      }
      return value.toFixed(4);
    }
    return value;
  };

  const getMetricColor = (metric, value) => {
    if (typeof value !== 'number') return 'text-gray-900';
    
    // Для метрик, где больше = лучше
    if (metric.includes('accuracy') || metric.includes('precision') || metric.includes('recall') || 
        metric.includes('f1') || metric.includes('sharpe') || metric.includes('profit')) {
      if (value >= 0.8) return 'text-green-600';
      if (value >= 0.6) return 'text-yellow-600';
      return 'text-red-600';
    }
    
    // Для метрик, где меньше = лучше
    if (metric.includes('mae') || metric.includes('mse') || metric.includes('rmse') || 
        metric.includes('mape') || metric.includes('drawdown')) {
      if (value <= 0.1) return 'text-green-600';
      if (value <= 0.3) return 'text-yellow-600';
      return 'text-red-600';
    }
    
    return 'text-gray-900';
  };

  const renderMetricsGrid = () => {
    if (!metrics && !results?.metrics) return null;
    
    const metricsData = results?.metrics || metrics;
    
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(metricsData).map(([metric, value]) => (
          <div key={metric} className="bg-white p-4 rounded-lg border shadow-sm">
            <div className="text-sm text-gray-600 capitalize mb-1">
              {metric.replace(/_/g, ' ')}
            </div>
            <div className={`text-2xl font-bold ${getMetricColor(metric, value)}`}>
              {formatMetric(metric, value)}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderCharts = () => {
    if (!results?.charts) return null;
    
    return (
      <div className="space-y-6">
        <h3 className="text-lg font-semibold text-gray-900">Графики</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {results.charts.map((chart, index) => (
            <div key={index} className="bg-white p-4 rounded-lg border shadow-sm">
              <div className="text-sm font-medium text-gray-900 mb-3">{chart.title}</div>
              <div className="h-64 bg-gray-50 rounded border-2 border-dashed border-gray-300 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-gray-400 text-4xl mb-2">📊</div>
                  <div className="text-gray-500 text-sm">{chart.type} Chart</div>
                  <div className="text-gray-400 text-xs mt-1">Chart visualization</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderRecommendations = () => {
    if (!results?.recommendations) return null;
    
    return (
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Рекомендации</h3>
        <div className="space-y-3">
          {results.recommendations.map((rec, index) => (
            <div key={index} className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
              <p className="text-gray-700">{rec}</p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderTrainingProgress = () => {
    if (!results?.training_progress) return null;
    
    const progress = results.training_progress;
    
    return (
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Прогресс обучения</h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Эпоха {progress.current_epoch}/{progress.total_epochs}</span>
              <span>{Math.round((progress.current_epoch / progress.total_epochs) * 100)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(progress.current_epoch / progress.total_epochs) * 100}%` }}
              ></div>
            </div>
          </div>
          
          {progress.loss && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-sm text-gray-600">Текущий loss</div>
                <div className="text-lg font-semibold text-gray-900">{progress.loss.toFixed(4)}</div>
              </div>
              {progress.val_loss && (
                <div>
                  <div className="text-sm text-gray-600">Validation loss</div>
                  <div className="text-lg font-semibold text-gray-900">{progress.val_loss.toFixed(4)}</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderConfusionMatrix = () => {
    if (!results?.confusion_matrix) return null;
    
    const matrix = results.confusion_matrix;
    const labels = matrix.labels || ['Buy', 'Hold', 'Sell'];
    
    return (
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Матрица ошибок</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Предсказано</th>
                {labels.map((label, index) => (
                  <th key={index} className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.data.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  <td className="px-3 py-2 text-sm font-medium text-gray-900 border-t">
                    {labels[rowIndex]}
                  </td>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} className="px-3 py-2 text-sm text-center border-t">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        rowIndex === cellIndex 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {cell}
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Загрузка результатов...</span>
      </div>
    );
  }

  if (!results && !metrics) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-400 text-4xl mb-4">📈</div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">Нет результатов</h3>
        <p className="text-gray-500">Запустите тестирование или обучение для получения результатов</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Метрики */}
      {renderMetricsGrid()}
      
      {/* Прогресс обучения */}
      {renderTrainingProgress()}
      
      {/* Матрица ошибок */}
      {renderConfusionMatrix()}
      
      {/* Графики */}
      {renderCharts()}
      
      {/* Рекомендации */}
      {renderRecommendations()}
    </div>
  );
}
