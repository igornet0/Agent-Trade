import React, { useEffect, useMemo, useState } from 'react';
import { get_agents, get_coins } from '../../services/strategyService';
import { evaluateModel, getTaskStatus } from '../../services/mlService';

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
  const [agentType, setAgentType] = useState('AgentPredTime');
  const [selectedAgentId, setSelectedAgentId] = useState('');
  const [selectedCoinIds, setSelectedCoinIds] = useState([]);
  const [timeframe, setTimeframe] = useState('5m');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [taskId, setTaskId] = useState(null);
  const [task, setTask] = useState(null);
  const [polling, setPolling] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const [ag, cs] = await Promise.all([
          get_agents('open'),
          get_coins(),
        ]);
        setAgents(ag);
        setCoins(cs);
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
        if (!t.ready) return; // keep until ready
        setPolling(false);
        clearInterval(timer);
      } catch (e) {
        console.error(e);
      }
    };
    setPolling(true);
    timer = setInterval(tick, 2000);
    return () => clearInterval(timer);
  }, [taskId]);

  const toggleCoin = (id) => {
    setSelectedCoinIds((prev) => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const onRun = async () => {
    try {
      const payload = {
        agent_id: Number(selectedAgentId) || undefined,
        coins: selectedCoinIds,
        timeframe,
        start: start || null,
        end: end || null,
      };
      const res = await evaluateModel(agentType, payload);
      setTaskId(res.task_id);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="lg:col-span-3">
      <div className="bg-white rounded-2xl shadow-lg p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-800">Module Tester</h2>
          <button
            onClick={onRun}
            disabled={(!selectedAgentId && agentType !== 'AgentNews') || selectedCoinIds.length === 0}
            className={`px-4 py-2 rounded-md font-medium ${((!selectedAgentId && agentType !== 'AgentNews') || selectedCoinIds.length === 0) ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
          >
            Запустить оценку
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Тип модуля</label>
            <select value={agentType} onChange={(e) => setAgentType(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
              {AGENT_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Агент</label>
            <select value={selectedAgentId} onChange={(e) => setSelectedAgentId(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
              <option value="">{agentType === 'AgentNews' ? 'Не требуется' : 'Выберите агента'}</option>
              {agents
                .filter(a => !agentType || a.type === agentType)
                .map(a => (
                <option value={a.id} key={a.id}>{a.name} ({a.type})</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Таймфрейм</label>
            <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
              {['5m','15m','30m','1h','4h','1d'].map(tf => <option key={tf} value={tf}>{tf}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Период</label>
            <div className="flex space-x-2">
              <input type="datetime-local" value={start} onChange={(e)=>setStart(e.target.value)} className="w-1/2 p-3 border border-gray-300 rounded-lg" />
              <input type="datetime-local" value={end} onChange={(e)=>setEnd(e.target.value)} className="w-1/2 p-3 border border-gray-300 rounded-lg" />
            </div>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Монеты</label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 max-h-64 overflow-auto p-2 border rounded-lg">
            {coins.map(c => (
              <label key={c.id} className="flex items-center space-x-2">
                <input type="checkbox" checked={selectedCoinIds.includes(c.id)} onChange={()=>toggleCoin(c.id)} />
                <span>{c.name}</span>
              </label>
            ))}
          </div>
        </div>

        {taskId && (
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-sm text-gray-600">Task: {taskId}</div>
            <div className="mt-2">
              {task ? (
                <div className="text-sm">
                  <div>Состояние: <span className="font-medium">{task.state}</span></div>

                  {/* Structured metrics view by module type */}
                  <MetricsView agentType={agentType} task={task} />

                  {/* Raw data fallback */}
                  {task.meta && task.meta.metrics && (
                    <details className="mt-3">
                      <summary className="cursor-pointer text-gray-700">Raw metrics</summary>
                      <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-auto max-h-64">{JSON.stringify(task.meta.metrics, null, 2)}</pre>
                    </details>
                  )}
                  {task.ready && task.successful && task.meta && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-gray-700">Raw meta</summary>
                      <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-auto max-h-64">{JSON.stringify(task.meta, null, 2)}</pre>
                    </details>
                  )}
                </div>
              ) : (
                <div className="flex items-center text-gray-600"><span className="animate-spin h-4 w-4 border-t-2 border-blue-500 rounded-full mr-2"/>Ожидание...</div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

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

  return null;
}

