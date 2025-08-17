import React, { useEffect, useMemo, useState } from 'react';
import { get_agents, get_coins, get_available_features } from '../../services/strategyService';
import { evaluateModel, getTaskStatus } from '../../services/mlService';
import TaskProgressWidget from './TaskProgressWidget';

const AGENT_TYPES = [
  { value: 'AgentPredTime', label: 'Pred_time' },
  { value: 'AgentTradeTime', label: 'Trade_time' },
  { value: 'AgentNews', label: 'News' },
  { value: 'AgentRisk', label: 'Risk' },
  { value: 'AgentTradeAggregator', label: 'Trade (Aggregator)' },
];

const ModuleTester = () => {
  const [agents, setAgents] = useState([]);
  const [coins, setCoins] = useState([]);
  const [features, setFeatures] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [evaluationConfig, setEvaluationConfig] = useState({});
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [agentsData, coinsData, featuresData] = await Promise.all([
        get_agents(),
        get_coins(),
        get_available_features()
      ]);
      setAgents(agentsData);
      setCoins(coinsData);
      setFeatures(featuresData);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    if (!selectedAgent) {
      alert('Выберите агента для оценки');
      return;
    }

    try {
      setLoading(true);
      const response = await evaluateModel(selectedAgent.type, {
        agent_id: selectedAgent.id,
        ...evaluationConfig
      });
      
      if (response.task_id) {
        setCurrentTaskId(response.task_id);
      } else {
        // Direct result
        setEvaluationResults(response);
      }
    } catch (error) {
      console.error('Error evaluating agent:', error);
      alert('Ошибка при оценке агента');
    } finally {
      setLoading(false);
    }
  };

  const handleTaskComplete = (taskData) => {
    setEvaluationResults(taskData.meta?.result || taskData);
    setCurrentTaskId(null);
  };

  const handleTaskError = (taskData) => {
    alert(`Ошибка оценки: ${taskData.error || 'Неизвестная ошибка'}`);
    setCurrentTaskId(null);
  };

  const getAgentTypeLabel = (type) => {
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

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Загрузка данных...</span>
      </div>
    );
  }

  return (
    <div className="lg:col-span-3">
      <div className="bg-white rounded-2xl shadow-lg p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-800">Module Tester</h2>
          <button
            onClick={handleEvaluate}
            disabled={loading}
            className={`px-4 py-2 rounded-md font-medium ${loading ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
          >
            {loading ? 'Запуск оценки...' : 'Запустить оценку'}
          </button>
        </div>

        {/* Agent Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Выберите агента для тестирования
          </label>
          <select
            value={selectedAgent?.id || ''}
            onChange={(e) => {
              const agent = agents.find(a => a.id === parseInt(e.target.value));
              setSelectedAgent(agent);
              setEvaluationResults(null);
            }}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">Выберите агента...</option>
            {agents.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.name} ({getAgentTypeLabel(agent.type)}) - {agent.status}
              </option>
            ))}
          </select>
        </div>

        {/* Evaluation Configuration */}
        {selectedAgent && (
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="text-lg font-medium text-gray-900 mb-3">
              Конфигурация оценки для {selectedAgent.name}
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Монеты для тестирования
                </label>
                <select
                  multiple
                  value={evaluationConfig.coins || []}
                  onChange={(e) => {
                    const selected = Array.from(e.target.selectedOptions, option => option.value);
                    setEvaluationConfig(prev => ({ ...prev, coins: selected }));
                  }}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                >
                  {coins.map((coin) => (
                    <option key={coin.id} value={coin.id}>
                      {coin.name} ({coin.symbol})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Таймфрейм
                </label>
                <select
                  value={evaluationConfig.timeframe || '5m'}
                  onChange={(e) => setEvaluationConfig(prev => ({ ...prev, timeframe: e.target.value }))}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="5m">5 минут</option>
                  <option value="15m">15 минут</option>
                  <option value="30m">30 минут</option>
                  <option value="1h">1 час</option>
                  <option value="4h">4 часа</option>
                  <option value="1d">1 день</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Период тестирования (дни)
                </label>
                <input
                  type="number"
                  min="1"
                  max="365"
                  value={evaluationConfig.test_period || 30}
                  onChange={(e) => setEvaluationConfig(prev => ({ ...prev, test_period: parseInt(e.target.value) }))}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                  placeholder="30"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Дополнительные параметры
                </label>
                <textarea
                  value={evaluationConfig.extra_params || ''}
                  onChange={(e) => setEvaluationConfig(prev => ({ ...prev, extra_params: e.target.value }))}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                  placeholder="JSON параметры для специфичных настроек"
                  rows="3"
                />
              </div>
            </div>

            <div className="mt-4">
              <button
                onClick={handleEvaluate}
                disabled={loading}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Запуск оценки...' : 'Запустить оценку'}
              </button>
            </div>
          </div>
        )}

        {/* Task Progress */}
        {currentTaskId && (
          <div className="mb-6">
            <h3 className="text-lg font-medium text-gray-900 mb-3">Прогресс оценки</h3>
            <TaskProgressWidget
              taskId={currentTaskId}
              onComplete={handleTaskComplete}
              onError={handleTaskError}
              autoStart={true}
            />
          </div>
        )}

        {/* Evaluation Results */}
        {evaluationResults && (
          <div className="mb-6">
            <h3 className="text-lg font-medium text-gray-900 mb-3">Результаты оценки</h3>
            <MetricsView agentType={selectedAgent?.type} task={{ meta: { metrics: evaluationResults } }} />
          </div>
        )}
      </div>
    </div>
  );
};

export default ModuleTester;


function MetricsView({ agentType, task }) {
  const meta = task?.meta || {};
  const metrics = meta?.metrics || meta || {};

  if (!metrics || typeof metrics !== 'object') return null;

  if (agentType === 'AgentPredTime') {
    const avgLoss = metrics.avg_loss ?? metrics.loss ?? meta.avg_loss;
    const samples = metrics.samples ?? metrics.sample_count;
    const coins = Array.isArray(metrics.coins) ? metrics.coins : [];
    const rmse = metrics.rmse;
    const mae = metrics.mae;
    const mape = metrics.mape;
    const directionAccuracy = metrics.direction_accuracy;
    
    return (
      <div className="mt-3 space-y-3">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {avgLoss !== undefined && (
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="text-xs text-blue-600 font-medium">Средняя ошибка</div>
              <div className="text-lg font-bold text-blue-800">{Number(avgLoss).toFixed(6)}</div>
            </div>
          )}
          {rmse !== undefined && (
            <div className="bg-green-50 p-3 rounded-lg">
              <div className="text-xs text-green-600 font-medium">RMSE</div>
              <div className="text-lg font-bold text-green-800">{Number(rmse).toFixed(6)}</div>
            </div>
          )}
          {mae !== undefined && (
            <div className="bg-amber-50 p-3 rounded-lg">
              <div className="text-xs text-amber-600 font-medium">MAE</div>
              <div className="text-lg font-bold text-amber-800">{Number(mae).toFixed(6)}</div>
            </div>
          )}
          {mape !== undefined && (
            <div className="bg-purple-50 p-3 rounded-lg">
              <div className="text-xs text-purple-600 font-medium">MAPE</div>
              <div className="text-lg font-bold text-purple-800">{Number(mape).toFixed(2)}%</div>
            </div>
          )}
        </div>
        
        {directionAccuracy !== undefined && (
          <div className="bg-indigo-50 p-3 rounded-lg">
            <div className="text-xs text-indigo-600 font-medium">Точность направления</div>
            <div className="text-lg font-bold text-indigo-800">{(Number(directionAccuracy) * 100).toFixed(2)}%</div>
          </div>
        )}
        
        {samples !== undefined && (
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="text-xs text-gray-600 font-medium">Количество образцов</div>
            <div className="text-lg font-bold text-gray-800">{samples}</div>
          </div>
        )}
        
        {coins.length > 0 && (
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-sm font-medium text-gray-700 mb-3">Метрики по монетам</div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Монета</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Средняя ошибка</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Образцы</th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">RMSE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {coins.map((c, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="px-3 py-2 font-medium text-gray-900">{c.coin_id ?? c.id ?? '-'}</td>
                      <td className="px-3 py-2 text-gray-700">{c.avg_loss !== undefined ? Number(c.avg_loss).toFixed(6) : '-'}</td>
                      <td className="px-3 py-2 text-gray-700">{c.samples ?? '-'}</td>
                      <td className="px-3 py-2 text-gray-700">{c.rmse !== undefined ? Number(c.rmse).toFixed(6) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (agentType === 'AgentTradeTime') {
    const { precision, recall, f1, accuracy, confusion, roc_auc, pr_auc } = metrics;
    
    return (
      <div className="mt-3 space-y-3">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {accuracy !== undefined && (
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="text-xs text-blue-600 font-medium">Точность</div>
              <div className="text-lg font-bold text-blue-800">{(Number(accuracy) * 100).toFixed(2)}%</div>
            </div>
          )}
          {precision !== undefined && (
            <div className="bg-amber-50 p-3 rounded-lg">
              <div className="text-xs text-amber-600 font-medium">Precision</div>
              <div className="text-lg font-bold text-amber-800">{(Number(precision) * 100).toFixed(2)}%</div>
            </div>
          )}
          {recall !== undefined && (
            <div className="bg-green-50 p-3 rounded-lg">
              <div className="text-xs text-green-600 font-medium">Recall</div>
              <div className="text-lg font-bold text-green-800">{(Number(recall) * 100).toFixed(2)}%</div>
            </div>
          )}
          {f1 !== undefined && (
            <div className="bg-purple-50 p-3 rounded-lg">
              <div className="text-xs text-purple-600 font-medium">F1-Score</div>
              <div className="text-lg font-bold text-purple-800">{(Number(f1) * 100).toFixed(2)}%</div>
            </div>
          )}
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          {roc_auc !== undefined && (
            <div className="bg-indigo-50 p-3 rounded-lg">
              <div className="text-xs text-indigo-600 font-medium">ROC-AUC</div>
              <div className="text-lg font-bold text-indigo-800">{Number(roc_auc).toFixed(4)}</div>
            </div>
          )}
          {pr_auc !== undefined && (
            <div className="bg-pink-50 p-3 rounded-lg">
              <div className="text-xs text-pink-600 font-medium">PR-AUC</div>
              <div className="text-lg font-bold text-pink-800">{Number(pr_auc).toFixed(4)}</div>
            </div>
          )}
        </div>
        
        {confusion && Array.isArray(confusion) && (
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-sm font-medium text-gray-700 mb-3">Матрица ошибок</div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase">Действие</th>
                    <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase">Hold</th>
                    <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase">Buy</th>
                    <th className="px-3 py-2 text-center text-xs font-medium text-gray-500 uppercase">Sell</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {['Hold', 'Buy', 'Sell'].map((action, i) => (
                    <tr key={action}>
                      <td className="px-3 py-2 text-center font-medium text-gray-900">{action}</td>
                      {confusion[i]?.map((value, j) => (
                        <td key={j} className="px-3 py-2 text-center text-gray-700">{value}</td>
                      )) || Array(3).fill('-').map((_, j) => (
                        <td key={j} className="px-3 py-2 text-center text-gray-700">-</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (agentType === 'AgentRisk') {
    const { avg_risk, max_drawdown, value_at_risk, expected_shortfall, violation_rate } = metrics;
    
    return (
      <div className="mt-3 space-y-3">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {avg_risk !== undefined && (
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="text-xs text-blue-600 font-medium">Средний риск</div>
              <div className="text-lg font-bold text-blue-800">{(Number(avg_risk) * 100).toFixed(2)}%</div>
            </div>
          )}
          {max_drawdown !== undefined && (
            <div className="bg-red-50 p-3 rounded-lg">
              <div className="text-xs text-red-600 font-medium">Макс. просадка</div>
              <div className="text-lg font-bold text-red-800">{(Number(max_drawdown) * 100).toFixed(2)}%</div>
            </div>
          )}
          {value_at_risk !== undefined && (
            <div className="bg-amber-50 p-3 rounded-lg">
              <div className="text-xs text-amber-600 font-medium">VaR (95%)</div>
              <div className="text-lg font-bold text-amber-800">{(Number(value_at_risk) * 100).toFixed(2)}%</div>
            </div>
          )}
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          {expected_shortfall !== undefined && (
            <div className="bg-purple-50 p-3 rounded-lg">
              <div className="text-xs text-purple-600 font-medium">Expected Shortfall</div>
              <div className="text-lg font-bold text-purple-800">{(Number(expected_shortfall) * 100).toFixed(2)}%</div>
            </div>
          )}
          {violation_rate !== undefined && (
            <div className="bg-indigo-50 p-3 rounded-lg">
              <div className="text-xs text-indigo-600 font-medium">Частота нарушений</div>
              <div className="text-lg font-bold text-indigo-800">{(Number(violation_rate) * 100).toFixed(2)}%</div>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (agentType === 'AgentTradeAggregator') {
    const { Sharpe, PnL, win_rate, max_drawdown, turnover_rate, exposure_stats } = metrics;
    
    return (
      <div className="mt-3 space-y-3">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Sharpe !== undefined && (
            <div className="bg-purple-50 p-3 rounded-lg">
              <div className="text-xs text-purple-600 font-medium">Коэффициент Шарпа</div>
              <div className="text-lg font-bold text-purple-800">{Number(Sharpe).toFixed(3)}</div>
            </div>
          )}
          {PnL !== undefined && (
            <div className="bg-green-50 p-3 rounded-lg">
              <div className="text-xs text-green-600 font-medium">PnL</div>
              <div className="text-lg font-bold text-green-800">{Number(PnL).toFixed(2)}</div>
            </div>
          )}
          {win_rate !== undefined && (
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="text-xs text-blue-600 font-medium">Процент выигрышей</div>
              <div className="text-lg font-bold text-blue-800">{(Number(win_rate) * 100).toFixed(2)}%</div>
            </div>
          )}
          {max_drawdown !== undefined && (
            <div className="bg-red-50 p-3 rounded-lg">
              <div className="text-xs text-red-600 font-medium">Макс. просадка</div>
              <div className="text-lg font-bold text-red-800">{(Number(max_drawdown) * 100).toFixed(2)}%</div>
            </div>
          )}
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          {turnover_rate !== undefined && (
            <div className="bg-amber-50 p-3 rounded-lg">
              <div className="text-xs text-amber-600 font-medium">Оборот</div>
              <div className="text-lg font-bold text-amber-800">{(Number(turnover_rate) * 100).toFixed(2)}%</div>
            </div>
          )}
          {exposure_stats && (
            <div className="bg-indigo-50 p-3 rounded-lg">
              <div className="text-xs text-indigo-600 font-medium">Статистика экспозиции</div>
              <div className="text-sm font-bold text-indigo-800">
                {typeof exposure_stats === 'object' ? 
                  `Средняя: ${(exposure_stats.avg || 0).toFixed(2)}%` : 
                  'Доступна'
                }
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (agentType === 'AgentNews') {
    const updated = metrics.updated ?? metrics.coins?.length;
    const correlation = metrics.correlation;
    const sentimentAccuracy = metrics.sentiment_accuracy;
    
    return (
      <div className="mt-3 space-y-3">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {updated !== undefined && (
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="text-xs text-blue-600 font-medium">Монет обновлено</div>
              <div className="text-lg font-bold text-blue-800">{updated}</div>
            </div>
          )}
          {correlation !== undefined && (
            <div className="bg-green-50 p-3 rounded-lg">
              <div className="text-xs text-green-600 font-medium">Корреляция</div>
              <div className="text-lg font-bold text-green-800">{Number(correlation).toFixed(4)}</div>
            </div>
          )}
          {sentimentAccuracy !== undefined && (
            <div className="bg-purple-50 p-3 rounded-lg">
              <div className="text-xs text-purple-600 font-medium">Точность сентимента</div>
              <div className="text-lg font-bold text-purple-800">{sentimentAccuracy}</div>
            </div>
          )}
        </div>
        
        {meta.progress !== undefined && (
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="text-xs text-gray-600 font-medium">Прогресс обработки</div>
            <div className="text-lg font-bold text-gray-800">{meta.progress}%</div>
          </div>
        )}
      </div>
    );
  }

  // Fallback for unknown agent types
  return (
    <div className="mt-3 p-4 bg-gray-50 rounded-lg">
      <div className="text-sm font-medium text-gray-700 mb-2">Метрики</div>
      <div className="text-sm text-gray-600">
        {Object.entries(metrics).map(([key, value]) => (
          <div key={key} className="flex justify-between py-1">
            <span className="font-medium">{key}:</span>
            <span className="font-mono">
              {typeof value === 'number' ? value.toFixed(4) : String(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

