import React, { useEffect, useState } from 'react';
import { savePipeline, runPipeline } from '../../services/pipelineService';
import { getTaskStatus } from '../../services/mlService';

export default function PipelineBuilder() {
  const [nodes, setNodes] = useState([
    { id: 'data', type: 'DataSource', config: { source: 'OHLCV' } },
    { id: 'pred', type: 'Pred_time', config: { seq_len: 96, pred_len: 12 } },
  ]);
  const [edges, setEdges] = useState([{ id: 'e1', source: 'data', target: 'pred' }]);
  const [timeframe, setTimeframe] = useState('5m');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [taskId, setTaskId] = useState(null);
  const [task, setTask] = useState(null);

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

  const config = { nodes, edges, timeframe, start: start || null, end: end || null };

  const onSave = async () => {
    try { await savePipeline(config); alert('Пайплайн сохранён'); } catch (e) { console.error(e); alert('Ошибка сохранения'); }
  };
  const onRun = async () => {
    try { const res = await runPipeline(config); setTaskId(res.task_id); } catch (e) { console.error(e); alert('Ошибка запуска'); }
  };

  return (
    <div className="lg:col-span-3">
      <div className="bg-white rounded-2xl shadow-lg p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-800">Pipeline Builder</h2>
          <div className="space-x-2">
            <button onClick={onSave} className="px-4 py-2 rounded-md font-medium bg-gray-100 hover:bg-gray-200">Сохранить</button>
            <button onClick={onRun} className="px-4 py-2 rounded-md font-medium bg-blue-600 text-white hover:bg-blue-700">Запустить</button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Таймфрейм</label>
            <select value={timeframe} onChange={(e)=>setTimeframe(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
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

        <div className="bg-gray-50 p-4 rounded-lg">
          <div className="text-sm text-gray-700 mb-2">Узлы</div>
          <pre className="text-xs bg-white p-3 rounded border overflow-auto max-h-64">{JSON.stringify(nodes, null, 2)}</pre>
          <div className="text-sm text-gray-700 mt-4 mb-2">Рёбра</div>
          <pre className="text-xs bg-white p-3 rounded border overflow-auto max-h-64">{JSON.stringify(edges, null, 2)}</pre>
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
      </div>
    </div>
  );
}


