import React, { useEffect, useState } from 'react';
import { testModel, getModelMetrics, getTaskStatus } from '../../services/mlService';
import { get_coins } from '../../services/strategyService';

const TEST_METRICS = {
  AgentNews: ['accuracy', 'precision', 'recall', 'f1_score', 'sentiment_accuracy'],
  AgentPredTime: ['mae', 'mse', 'rmse', 'mape', 'directional_accuracy'],
  AgentTradeTime: ['accuracy', 'precision', 'recall', 'f1_score', 'profit_factor'],
  AgentRisk: ['risk_accuracy', 'max_drawdown', 'sharpe_ratio', 'calmar_ratio'],
  AgentTradeAggregator: ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
};

const TIMEFRAMES = ['5m','15m','30m','1h','4h','1d'];

export default function ModelTester() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [coins, setCoins] = useState([]);
  const [selectedCoinIds, setSelectedCoinIds] = useState([]);
  const [timeframe, setTimeframe] = useState('5m');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [testMetrics, setTestMetrics] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [task, setTask] = useState(null);
  const [results, setResults] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [modelsData, coinsData] = await Promise.all([
          getModels(),
          get_coins()
        ]);
        setModels(modelsData);
        setCoins(coinsData);
      } catch (e) {
        console.error(e);
      }
    };
    load();
  }, []);

  useEffect(() => {
    if (!taskId) return;
    let timer;
    const tick = async () => {
      try {
        const t = await getTaskStatus(taskId);
        setTask(t);
        if (t.ready) {
          clearInterval(timer);
          if (t.result) {
            setResults(t.result);
          }
        }
      } catch (e) { console.error(e); }
    };
    timer = setInterval(tick, 2000);
    return () => clearInterval(timer);
  }, [taskId]);

  // Обновляем метрики при изменении модели
  useEffect(() => {
    if (selectedModel) {
      const modelType = selectedModel.type;
      setTestMetrics(TEST_METRICS[modelType] || []);
    }
  }, [selectedModel]);

  const getModels = async () => {
    // TODO: Реализовать API для получения списка моделей
    return [];
  };

  const toggleCoin = (id) => {
    setSelectedCoinIds((prev) => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const onTest = async () => {
    if (!selectedModel || selectedCoinIds.length === 0) return;
    setSubmitting(true);
    try {
      const payload = {
        model_id: selectedModel.id,
        coins: selectedCoinIds,
        timeframe,
        start_date: startDate,
        end_date: endDate,
        metrics: testMetrics
      };

      const res = await testModel(payload);
      if (res && res.task_id) setTaskId(res.task_id);
      alert('Тестирование запущено');
    } catch (e) {
      console.error(e);
      alert('Ошибка запуска тестирования');
    } finally {
      setSubmitting(false);
    }
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

  const formatMetric = (metric, value) => {
    if (typeof value === 'number') {
      if (metric.includes('ratio') || metric.includes('rate') || metric.includes('accuracy')) {
        return `${(value * 100).toFixed(2)}%`;
      }
      return value.toFixed(4);
    }
    return value;
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Тестирование модели</h2>
          <p className="text-sm text-gray-600 mt-1">Оценка производительности ML-моделей на исторических данных</p>
        </div>
        <button 
          onClick={onTest} 
          disabled={submitting || !selectedModel || selectedCoinIds.length === 0}
          className={`px-4 py-2 rounded-md font-medium ${
            (!selectedModel || selectedCoinIds.length === 0) 
              ? 'bg-gray-300 cursor-not-allowed' 
              : 'bg-green-600 text-white hover:bg-green-700'
          }`}
        >
          {submitting ? 'Запуск...' : 'Запустить тест'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Выберите модель</label>
          <select 
            value={selectedModel?.id || ''} 
            onChange={(e) => {
              const model = models.find(m => m.id === Number(e.target.value));
              setSelectedModel(model);
            }} 
            className="w-full p-3 border border-gray-300 rounded-lg"
          >
            <option value="">Выберите модель для тестирования</option>
            {models.map(model => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.type})
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Таймфрейм</label>
          <select 
            value={timeframe} 
            onChange={(e)=>setTimeframe(e.target.value)} 
            className="w-full p-3 border border-gray-300 rounded-lg"
          >
            {TIMEFRAMES.map(tf => <option key={tf} value={tf}>{tf}</option>)}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Начальная дата</label>
          <input 
            type="date" 
            value={startDate} 
            onChange={(e)=>setStartDate(e.target.value)} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Конечная дата</label>
          <input 
            type="date" 
            value={endDate} 
            onChange={(e)=>setEndDate(e.target.value)} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
          />
        </div>
      </div>

      {selectedModel && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Информация о модели</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <span className="text-sm text-gray-600">Название:</span>
              <p className="font-medium">{selectedModel.name}</p>
            </div>
            <div>
              <span className="text-sm text-gray-600">Тип:</span>
              <p className="font-medium">{selectedModel.type}</p>
            </div>
            <div>
              <span className="text-sm text-gray-600">Статус:</span>
              <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(selectedModel.status)}`}>
                {selectedModel.status}
              </span>
            </div>
            {selectedModel.timeframe && (
              <div>
                <span className="text-sm text-gray-600">Таймфрейм:</span>
                <p className="font-medium">{selectedModel.timeframe}</p>
              </div>
            )}
            {selectedModel.version && (
              <div>
                <span className="text-sm text-gray-600">Версия:</span>
                <p className="font-medium">{selectedModel.version}</p>
              </div>
            )}
            {selectedModel.created_at && (
              <div>
                <span className="text-sm text-gray-600">Создана:</span>
                <p className="font-medium">{new Date(selectedModel.created_at).toLocaleDateString()}</p>
              </div>
            )}
          </div>
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Монеты для тестирования</label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 max-h-64 overflow-auto p-2 border rounded-lg">
          {coins.map(c => (
            <label key={c.id} className="flex items-center space-x-2">
              <input 
                type="checkbox" 
                checked={selectedCoinIds.includes(c.id)} 
                onChange={()=>toggleCoin(c.id)} 
              />
              <span>{c.name}</span>
            </label>
          ))}
        </div>
      </div>

      {testMetrics.length > 0 && (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Метрики для оценки</label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 p-2 border rounded-lg">
            {testMetrics.map(metric => (
              <label key={metric} className="flex items-center space-x-2">
                <input 
                  type="checkbox" 
                  checked={testMetrics.includes(metric)} 
                  onChange={(e) => {
                    if (e.target.checked) {
                      setTestMetrics(prev => [...prev, metric]);
                    } else {
                      setTestMetrics(prev => prev.filter(m => m !== metric));
                    }
                  }} 
                />
                <span className="text-sm">{metric}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {taskId && (
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600">Task: <span className="font-medium">{taskId}</span></div>
          <div className="mt-2">
            {task ? (
              <div className="text-sm">
                <div>Состояние: <span className="font-medium">{task.state}</span></div>
                {task.meta && (
                  <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-auto max-h-64">
                    {JSON.stringify(task.meta, null, 2)}
                  </pre>
                )}
              </div>
            ) : (
              <div className="flex items-center text-gray-600">
                <span className="animate-spin h-4 w-4 border-t-2 border-blue-500 rounded-full mr-2"/>
                Ожидание...
              </div>
            )}
          </div>
        </div>
      )}

      {results && (
        <div className="p-4 bg-green-50 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Результаты тестирования</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(results.metrics || {}).map(([metric, value]) => (
              <div key={metric} className="bg-white p-3 rounded border">
                <div className="text-sm text-gray-600 capitalize">{metric.replace(/_/g, ' ')}</div>
                <div className="text-lg font-semibold text-gray-900">
                  {formatMetric(metric, value)}
                </div>
              </div>
            ))}
          </div>
          
          {results.charts && (
            <div className="mt-6">
              <h4 className="text-md font-semibold text-gray-900 mb-2">Графики</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {results.charts.map((chart, index) => (
                  <div key={index} className="bg-white p-3 rounded border">
                    <div className="text-sm text-gray-600 mb-2">{chart.title}</div>
                    <div className="h-48 bg-gray-100 rounded flex items-center justify-center">
                      <span className="text-gray-500">График {chart.type}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {results.recommendations && (
            <div className="mt-6">
              <h4 className="text-md font-semibold text-gray-900 mb-2">Рекомендации</h4>
              <div className="bg-white p-3 rounded border">
                <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                  {results.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
