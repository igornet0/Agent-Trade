import React, { useEffect, useState } from 'react';
import { get_coins } from '../../services/strategyService';
import { recalcNewsBackground, getNewsBackground, getTaskStatus } from '../../services/mlService';

export default function NewsTrainPanel() {
  const [coins, setCoins] = useState([]);
  const [selectedCoinIds, setSelectedCoinIds] = useState([]);
  const [viewCoinId, setViewCoinId] = useState('');
  const [bg, setBg] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [task, setTask] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const cs = await get_coins();
        setCoins(cs);
      } catch (e) { console.error(e); }
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
        if (t.ready) clearInterval(timer);
      } catch (e) { console.error(e); }
    };
    timer = setInterval(tick, 2000);
    return () => clearInterval(timer);
  }, [taskId]);

  const toggleCoin = (id) => {
    setSelectedCoinIds((prev) => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const onRecalc = async () => {
    try {
      const res = await recalcNewsBackground({ coins: selectedCoinIds });
      if (res && res.task_id) setTaskId(res.task_id);
      alert('Пересчёт фона запущен');
    } catch (e) { console.error(e); alert('Ошибка запуска пересчёта'); }
  };

  const onView = async () => {
    if (!viewCoinId) return;
    try {
      const res = await getNewsBackground(viewCoinId);
      setBg(res);
    } catch (e) { console.error(e); setBg({ error: 'Нет данных' }); }
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-800">News Background</h2>
        <button onClick={onRecalc} disabled={selectedCoinIds.length === 0}
          className={`px-4 py-2 rounded-md font-medium ${selectedCoinIds.length === 0 ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
        >Пересчитать фон</button>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Монеты для пересчёта</label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 max-h-64 overflow-auto p-2 border rounded-lg">
          {coins.map(c => (
            <label key={c.id} className="flex items-center space-x-2">
              <input type="checkbox" checked={selectedCoinIds.includes(c.id)} onChange={()=>toggleCoin(c.id)} />
              <span>{c.name}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="md:col-span-3">
          <label className="block text-sm font-medium text-gray-700 mb-2">Просмотр фона по монете</label>
          <div className="flex space-x-2">
            <select value={viewCoinId} onChange={(e)=>setViewCoinId(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
              <option value="">Выберите монету</option>
              {coins.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
            <button onClick={onView} className="px-4 py-2 rounded-md font-medium bg-indigo-600 text-white hover:bg-indigo-700">Показать</button>
          </div>
        </div>
      </div>

      {taskId && (
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600">Task: <span className="font-medium">{taskId}</span></div>
          <div className="mt-2">
            {task ? (
              <div className="text-sm">
                <div>Состояние: <span className="font-medium">{task.state}</span></div>
                {task.meta && (
                  <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-auto max-h-64">{JSON.stringify(task.meta, null, 2)}</pre>
                )}
              </div>
            ) : (
              <div className="flex items-center text-gray-600"><span className="animate-spin h-4 w-4 border-t-2 border-blue-500 rounded-full mr-2"/>Ожидание...</div>
            )}
          </div>
        </div>
      )}

      {bg && (
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600">Фон для монеты {viewCoinId}:</div>
          <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-auto max-h-64">{JSON.stringify(bg, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}


