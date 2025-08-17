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
    return (
      <div className="mt-3 space-y-2">
        <div className="flex flex-wrap gap-3">
          {avgLoss !== undefined && (<span className="px-2 py-1 rounded bg-blue-100 text-blue-800 text-xs">avg_loss: {Number(avgLoss).toFixed(6)}</span>)}
          {samples !== undefined && (<span className="px-2 py-1 rounded bg-green-100 text-green-800 text-xs">samples: {samples}</span>)}
        </div>
        {coins.length > 0 && (
          <div className="mt-2">
            <div className="text-xs text-gray-600 mb-1">Метрики по монетам</div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs border rounded">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="px-2 py-1 text-left">coin_id</th>
                    <th className="px-2 py-1 text-left">avg_loss</th>
                    <th className="px-2 py-1 text-left">samples</th>
                  </tr>
                </thead>
                <tbody>
                  {coins.map((c, i) => (
                    <tr key={i} className="odd:bg-white even:bg-gray-50">
                      <td className="px-2 py-1">{c.coin_id ?? c.id ?? '-'}</td>
                      <td className="px-2 py-1">{c.avg_loss !== undefined ? Number(c.avg_loss).toFixed(6) : '-'}</td>
                      <td className="px-2 py-1">{c.samples ?? '-'}</td>
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
    const { precision, recall, f1, accuracy, confusion } = metrics;
    return (
      <div className="mt-3 space-y-2">
        <div className="flex flex-wrap gap-3">
          {accuracy !== undefined && (<span className="px-2 py-1 rounded bg-blue-100 text-blue-800 text-xs">accuracy: {Number(accuracy).toFixed(4)}</span>)}
          {precision !== undefined && (<span className="px-2 py-1 rounded bg-amber-100 text-amber-800 text-xs">precision: {Number(precision).toFixed(4)}</span>)}
          {recall !== undefined && (<span className="px-2 py-1 rounded bg-green-100 text-green-800 text-xs">recall: {Number(recall).toFixed(4)}</span>)}
          {f1 !== undefined && (<span className="px-2 py-1 rounded bg-purple-100 text-purple-800 text-xs">f1: {Number(f1).toFixed(4)}</span>)}
        </div>
        {confusion && Array.isArray(confusion) && (
          <div className="text-xs text-gray-600">confusion: {JSON.stringify(confusion)}</div>
        )}
      </div>
    );
  }

  if (agentType === 'AgentRisk') {
    const { avg_risk, max_drawdown, value_at_risk } = metrics;
    return (
      <div className="mt-3 flex flex-wrap gap-3">
        {avg_risk !== undefined && (<span className="px-2 py-1 rounded bg-blue-100 text-blue-800 text-xs">avg_risk: {Number(avg_risk).toFixed(4)}</span>)}
        {max_drawdown !== undefined && (<span className="px-2 py-1 rounded bg-red-100 text-red-800 text-xs">max_dd: {Number(max_drawdown).toFixed(4)}</span>)}
        {value_at_risk !== undefined && (<span className="px-2 py-1 rounded bg-amber-100 text-amber-800 text-xs">VaR: {Number(value_at_risk).toFixed(4)}</span>)}
      </div>
    );
  }

  if (agentType === 'AgentTradeAggregator') {
    const { Sharpe, PnL, win_rate } = metrics;
    return (
      <div className="mt-3 flex flex-wrap gap-3">
        {Sharpe !== undefined && (<span className="px-2 py-1 rounded bg-purple-100 text-purple-800 text-xs">Sharpe: {Sharpe}</span>)}
        {PnL !== undefined && (<span className="px-2 py-1 rounded bg-green-100 text-green-800 text-xs">PnL: {PnL}</span>)}
        {win_rate !== undefined && (<span className="px-2 py-1 rounded bg-blue-100 text-blue-800 text-xs">win_rate: {win_rate}</span>)}
      </div>
    );
  }

  if (agentType === 'AgentNews') {
    const updated = metrics.updated ?? metrics.coins?.length;
    return (
      <div className="mt-3 flex flex-wrap gap-3">
        {updated !== undefined && (<span className="px-2 py-1 rounded bg-blue-100 text-blue-800 text-xs">coins updated: {updated}</span>)}
        {meta.progress !== undefined && (<span className="px-2 py-1 rounded bg-gray-100 text-gray-800 text-xs">progress: {meta.progress}%</span>)}
      </div>
    );
  }

  return null;
}

