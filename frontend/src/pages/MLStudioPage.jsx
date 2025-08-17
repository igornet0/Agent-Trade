import React, { useState, useEffect } from 'react';
import { listModels, getTaskStatus } from '../services/mlService';
import { get_coins } from '../services/strategyService';

const MLStudioPage = () => {
  const [models, setModels] = useState([]);
  const [coins, setCoins] = useState([]);
  const [selectedType, setSelectedType] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTasks, setActiveTasks] = useState(new Map());

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [modelsData, coinsData] = await Promise.all([
        listModels(),
        get_coins()
      ]);
      setModels(modelsData);
      setCoins(coinsData);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const startTaskPolling = (taskId) => {
    if (activeTasks.has(taskId)) return; // Already polling
    
    const pollInterval = setInterval(async () => {
      try {
        const task = await getTaskStatus(taskId);
        setActiveTasks(prev => new Map(prev.set(taskId, task)));
        
        if (task.ready) {
          clearInterval(pollInterval);
          setActiveTasks(prev => {
            const newMap = new Map(prev);
            newMap.delete(taskId);
            return newMap;
          });
          
          // Refresh models list after task completion
          loadData();
        }
      } catch (error) {
        console.error(`Error polling task ${taskId}:`, error);
        clearInterval(pollInterval);
      }
    }, 2000);

    // Store interval reference for cleanup
    setActiveTasks(prev => new Map(prev.set(taskId, { state: 'PENDING', ready: false })));
  };

  const getModelTypeLabel = (type) => {
    const typeMap = {
      'AgentNews': 'News',
      'AgentPredTime': 'Pred_time',
      'AgentTradeTime': 'Trade_time',
      'AgentRisk': 'Risk',
      'AgentTradeAggregator': 'Trade (Aggregator)'
    };
    return typeMap[type] || type;
  };

  const getStatusColor = (status) => {
    const colorMap = {
      'open': 'bg-green-100 text-green-800',
      'training': 'bg-yellow-100 text-yellow-800',
      'error': 'bg-red-100 text-red-800',
      'completed': 'bg-blue-100 text-blue-800'
    };
    return colorMap[status] || 'bg-gray-100 text-gray-800';
  };

  const filteredModels = selectedType 
    ? models.filter(model => model.type === selectedType)
    : models;

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">ML Studio</h1>
          <p className="mt-2 text-gray-600">
            Управление и мониторинг машинного обучения моделей
          </p>
        </div>

        {/* Active Tasks Monitor */}
        {activeTasks.size > 0 && (
          <div className="mb-8 bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Активные задачи</h2>
            <div className="space-y-3">
              {Array.from(activeTasks.entries()).map(([taskId, task]) => (
                <div key={taskId} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                    <span className="text-sm font-medium text-gray-700">Task {taskId}</span>
                    <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(task.state?.toLowerCase())}`}>
                      {task.state || 'PENDING'}
                    </span>
                  </div>
                  {task.meta?.progress && (
                    <div className="flex items-center space-x-2">
                      <div className="w-32 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${task.meta.progress}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-600">{task.meta.progress}%</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="mb-6 bg-white rounded-lg shadow p-6">
          <div className="flex items-center space-x-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Тип модели
              </label>
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                className="p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Все типы</option>
                <option value="AgentNews">News</option>
                <option value="AgentPredTime">Pred_time</option>
                <option value="AgentTradeTime">Trade_time</option>
                <option value="AgentRisk">Risk</option>
                <option value="AgentTradeAggregator">Trade (Aggregator)</option>
              </select>
            </div>
            
            <div className="ml-auto">
              <button
                onClick={loadData}
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Обновление...' : 'Обновить'}
              </button>
            </div>
          </div>
        </div>

        {/* Models Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredModels.map((model) => (
            <div key={model.id} className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
                  <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(model.status)}`}>
                    {model.status}
                  </span>
                </div>
                
                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Тип:</span>
                    <span className="font-medium">{getModelTypeLabel(model.type)}</span>
                  </div>
                  
                  {model.timeframe && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Таймфрейм:</span>
                      <span className="font-medium">{model.timeframe}</span>
                    </div>
                  )}
                  
                  {model.created_at && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Создан:</span>
                      <span className="font-medium">
                        {new Date(model.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                  
                  {model.updated_at && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Обновлен:</span>
                      <span className="font-medium">
                        {new Date(model.updated_at).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                </div>

                <div className="flex space-x-2">
                  <button
                    onClick={() => {
                      // TODO: Implement model evaluation
                      console.log('Evaluate model:', model.id);
                    }}
                    className="flex-1 px-3 py-2 bg-green-600 text-white text-sm rounded-md hover:bg-green-700 transition-colors"
                  >
                    Оценить
                  </button>
                  
                  {model.status === 'completed' && (
                    <button
                      onClick={() => {
                        // TODO: Implement model promotion
                        console.log('Promote model:', model.id);
                      }}
                      className="flex-1 px-3 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors"
                    >
                      Продвинуть
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredModels.length === 0 && !loading && (
          <div className="text-center py-12">
            <div className="text-gray-400 text-6xl mb-4">🤖</div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Модели не найдены</h3>
            <p className="text-gray-500">
              {selectedType ? `Нет моделей типа "${getModelTypeLabel(selectedType)}"` : 'Создайте первую модель для начала работы'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MLStudioPage;
