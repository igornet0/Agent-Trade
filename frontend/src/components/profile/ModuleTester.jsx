import React, { useEffect, useMemo, useState } from 'react';
import { get_agents, get_coins, evaluate_agent, get_task_status } from '../../services/strategyService';

const ModuleTester = () => {
  const [agents, setAgents] = useState([]);
  const [coins, setCoins] = useState([]);
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
        const t = await get_task_status(taskId);
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
        agent_id: Number(selectedAgentId),
        coins: selectedCoinIds,
        timeframe,
        start: start || null,
        end: end || null,
      };
      const res = await evaluate_agent(payload);
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
            disabled={!selectedAgentId || selectedCoinIds.length === 0}
            className={`px-4 py-2 rounded-md font-medium ${(!selectedAgentId || selectedCoinIds.length === 0) ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
          >
            Запустить оценку
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Агент</label>
            <select value={selectedAgentId} onChange={(e) => setSelectedAgentId(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
              <option value="">Выберите агента</option>
              {agents.map(a => (
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
                  {task.meta && task.meta.metrics && (
                    <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-auto max-h-64">{JSON.stringify(task.meta.metrics, null, 2)}</pre>
                  )}
                  {task.ready && task.successful && task.meta && (
                    <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-auto max-h-64">{JSON.stringify(task.meta, null, 2)}</pre>
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


