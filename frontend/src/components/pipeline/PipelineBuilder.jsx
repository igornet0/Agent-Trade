import React, { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, { Background, Controls, addEdge, MiniMap, ReactFlowProvider, useReactFlow } from 'reactflow';
import 'reactflow/dist/style.css';
import { savePipeline, runPipeline, loadPipeline, revokePipeline, getBacktests, getBacktest } from '../../services/pipelineService';
import { getTaskStatus } from '../../services/mlService';

export default function PipelineBuilder() {
  const DEFAULT_CONFIGS = useMemo(() => ({
    DataSource: { source: 'OHLCV' },
    Indicators: { indicators: ['SMA(20)','RSI(14)','MACD'] },
    News: { model: 'finbert', window: 288, horizon: 12 },
    Pred_time: { model: 'LSTM', seq_len: 96, pred_len: 12 },
    Trade_time: { classifier: 'LightGBM', target: 'buy_sell_hold' },
    Risk: { model: 'XGBoost', max_leverage: 3, risk_limit_pct: 2 },
    Trade: { mode: 'rules', weights: { pred: 0.5, trade: 0.3, risk: 0.2 } },
    Metrics: { outputs: ['PnL','Sharpe','WinRate'] },
  }), []);

  const PRESET_VARIANTS = useMemo(() => ({
    News: [
      { label: 'FinBERT (по умолч.)', config: { model: 'finbert', window: 288, horizon: 12 } },
      { label: 'BERT', config: { model: 'bert', window: 288, horizon: 12 } },
    ],
    Pred_time: [
      { label: 'LSTM 96→12', config: { model: 'LSTM', seq_len: 96, pred_len: 12 } },
      { label: 'Transformer 192→24', config: { model: 'Transformer', seq_len: 192, pred_len: 24 } },
    ],
    Trade_time: [
      { label: 'LightGBM', config: { classifier: 'LightGBM', target: 'buy_sell_hold' } },
      { label: 'CatBoost', config: { classifier: 'CatBoost', target: 'buy_sell_hold' } },
    ],
    Risk: [
      { label: 'XGBoost базовый', config: { model: 'XGBoost', max_leverage: 3, risk_limit_pct: 2 } },
      { label: 'Эвристика', config: { model: 'heuristic', max_drawdown_pct: 20, per_trade_risk_pct: 1 } },
    ],
    Trade: [
      { label: 'Rules (веса пред/сигн/риск)', config: { mode: 'rules', weights: { pred: 0.5, trade: 0.3, risk: 0.2 } } },
      { label: 'RL (заглушка)', config: { mode: 'rl', gamma: 0.99, learning_rate: 3e-4 } },
    ],
  }), []);
  const [nodes, setNodes] = useState([
    { id: 'data', type: 'DataSource', position: { x: 100, y: 100 }, data: { label: 'DataSource', config: { source: 'OHLCV' } } },
    { id: 'pred', type: 'Pred_time', position: { x: 400, y: 100 }, data: { label: 'Pred_time', config: { seq_len: 96, pred_len: 12 } } },
  ]);
  const [edges, setEdges] = useState([{ id: 'e1', source: 'data', target: 'pred' }]);
  const [timeframe, setTimeframe] = useState('5m');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [taskId, setTaskId] = useState(null);
  const [task, setTask] = useState(null);
  const [savedPipelineId, setSavedPipelineId] = useState('');
  const [loadPipelineId, setLoadPipelineId] = useState('');
  const [importInputKey, setImportInputKey] = useState(() => String(Date.now()));
  const copyToClipboard = useCallback(async (text) => {
    try { await navigator.clipboard.writeText(text); alert('Скопировано в буфер обмена'); } catch (e) { console.error(e); }
  }, []);
  const [backtests, setBacktests] = useState([]);
  const [showBacktests, setShowBacktests] = useState(false);
  const refreshBacktests = useCallback(async ()=>{
    try { const data = await getBacktests(); setBacktests(data || []); } catch(e){ console.error(e);} 
  }, []);
  useEffect(() => { refreshBacktests(); }, [refreshBacktests]);

  // Simple palette / editor state
  const NODE_TYPES = useMemo(() => ['DataSource','Indicators','News','Pred_time','Trade_time','Risk','Trade','Metrics'], []);
  const [newNodeId, setNewNodeId] = useState('node1');
  const [newNodeType, setNewNodeType] = useState('DataSource');
  const [newNodeConfig, setNewNodeConfig] = useState('{"source":"OHLCV"}');

  const [newEdgeSource, setNewEdgeSource] = useState('data');
  const [newEdgeTarget, setNewEdgeTarget] = useState('pred');

  // Selection / right panel editor
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [selectedNodeType, setSelectedNodeType] = useState('DataSource');
  const [selectedNodeConfig, setSelectedNodeConfig] = useState('{}');

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

  // Serialize ReactFlow nodes -> PipelineConfig
  const pipelineConfig = useMemo(() => ({
    nodes: nodes.map(n => ({ id: n.id, type: n.type, config: n.data?.config || {} })),
    edges: edges.map(e => ({ id: e.id, source: e.source, target: e.target })),
    timeframe,
    start: start || null,
    end: end || null,
  }), [nodes, edges, timeframe, start, end]);

  const onSave = async () => {
    try {
      const res = await savePipeline(pipelineConfig);
      const pid = res.pipeline_id || '';
      setSavedPipelineId(pid);
      alert(`Пайплайн сохранён${pid ? ` (id: ${pid})` : ''}`);
    } catch (e) { console.error(e); alert('Ошибка сохранения'); }
  };
  const onRun = async () => {
    try { const res = await runPipeline(pipelineConfig); setTaskId(res.task_id); } catch (e) { console.error(e); alert('Ошибка запуска'); }
  };

  const onLoad = async () => {
    if (!loadPipelineId) return alert('Укажите pipeline id');
    try {
      const cfg = await loadPipeline(loadPipelineId);
      const rfNodes = (cfg.nodes || []).map((n, idx) => ({ id: n.id, type: n.type, position: { x: 120 + (idx%4)*180, y: 80 + Math.floor(idx/4)*120 }, data: { label: n.type, config: n.config || {} } }));
      const rfEdges = (cfg.edges || []).map((e, i) => ({ id: e.id || `e_${e.source}_${e.target}_${i+1}`, source: e.source, target: e.target }));
      setNodes(rfNodes);
      setEdges(rfEdges);
      setTimeframe(cfg.timeframe || '5m');
      setStart(cfg.start || '');
      setEnd(cfg.end || '');
      alert('Пайплайн загружен');
    } catch (e) { console.error(e); alert('Ошибка загрузки пайплайна'); }
  };

  const exportJson = () => {
    try {
      const dataStr = JSON.stringify(pipelineConfig, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pipeline_${new Date().toISOString().slice(0,19)}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error(e); alert('Ошибка экспорта');
    }
  };

  const onImportFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const cfg = JSON.parse(text);
      const rfNodes = (cfg.nodes || []).map((n, idx) => ({ id: n.id, type: n.type, position: { x: 120 + (idx%4)*180, y: 80 + Math.floor(idx/4)*120 }, data: { label: n.type, config: n.config || {} } }));
      const rfEdges = (cfg.edges || []).map((e, i) => ({ id: e.id || `e_${e.source}_${e.target}_${i+1}`, source: e.source, target: e.target }));
      setNodes(rfNodes);
      setEdges(rfEdges);
      setTimeframe(cfg.timeframe || '5m');
      setStart(cfg.start || '');
      setEnd(cfg.end || '');
      setImportInputKey(String(Date.now()));
      alert('Импортировано из файла');
    } catch (err) {
      console.error(err); alert('Некорректный JSON');
    }
  };

  const onAddNode = () => {
    if (!newNodeId) return alert('Укажите id узла');
    if (nodes.some(n => n.id === newNodeId)) return alert('Узел с таким id уже существует');
    let parsed = {};
    try { parsed = newNodeConfig ? JSON.parse(newNodeConfig) : {}; } catch { return alert('config должен быть валидным JSON'); }
    const baseCfg = Object.keys(parsed).length ? parsed : (DEFAULT_CONFIGS[newNodeType] || {});
    setNodes(prev => [...prev, { id: newNodeId, type: newNodeType, position: { x: 200, y: 200 + prev.length * 40 }, data: { label: newNodeType, config: baseCfg } }]);
  };

  const onRemoveNode = (id) => {
    setNodes(prev => prev.filter(n => n.id !== id));
    setEdges(prev => prev.filter(e => e.source !== id && e.target !== id));
  };

  const ALLOWED_NEXT = useMemo(() => ({
    DataSource: ['Indicators','News','Pred_time','Metrics'],
    Indicators: ['Pred_time','Trade_time','Metrics'],
    News: ['Pred_time','Trade_time','Risk','Metrics'],
    Pred_time: ['Trade_time','Metrics'],
    Trade_time: ['Risk','Trade','Metrics'],
    Risk: ['Trade','Metrics'],
    Trade: ['Metrics'],
    Metrics: [],
  }), []);

  const getNodeType = useCallback((id) => nodes.find(n => n.id === id)?.type, [nodes]);
  const validateConnection = useCallback((sourceId, targetId) => {
    const sType = getNodeType(sourceId);
    const tType = getNodeType(targetId);
    if (!sType || !tType) return false;
    const allowed = ALLOWED_NEXT[sType] || [];
    return allowed.includes(tType);
  }, [ALLOWED_NEXT, getNodeType]);

  const onAddEdge = () => {
    if (!newEdgeSource || !newEdgeTarget) return;
    if (newEdgeSource === newEdgeTarget) return alert('source и target не должны совпадать');
    if (!validateConnection(newEdgeSource, newEdgeTarget)) return alert('Невалидная связь для выбранных узлов');
    const id = `e_${newEdgeSource}_${newEdgeTarget}_${edges.length+1}`;
    setEdges(prev => [...prev, { id, source: newEdgeSource, target: newEdgeTarget }]);
  };

  const onRemoveEdge = (id) => setEdges(prev => prev.filter(e => e.id !== id));

  // ReactFlow handlers
  const onConnect = useCallback((params) => {
    if (!validateConnection(params.source, params.target)) {
      alert('Невалидная связь для выбранных типов узлов');
      return;
    }
    setEdges((eds) => addEdge({ ...params, id: `e_${params.source}_${params.target}_${eds.length+1}` }, eds));
  }, [validateConnection]);
  const onNodesChange = useCallback((changes) => {
    setNodes((nds) => nds.map(n => {
      const ch = changes.find(c => c.id === n.id);
      if (!ch) return n;
      if (ch.type === 'position') return { ...n, position: ch.position };
      return n;
    }));
  }, []);
  const onEdgesChange = useCallback(() => {}, []);
  const onNodeClick = useCallback((_, node) => {
    setSelectedNodeId(node.id);
    setSelectedNodeType(node.type);
    const cfg = node.data?.config || {};
    setSelectedNodeConfig(JSON.stringify(cfg, null, 2));
  }, []);

  const onUpdateSelectedNode = () => {
    if (!selectedNodeId) return;
    let parsed = {};
    try { parsed = selectedNodeConfig ? JSON.parse(selectedNodeConfig) : {}; } catch { return alert('config должен быть валидным JSON'); }
    setNodes(prev => prev.map(n => n.id === selectedNodeId ? { ...n, type: selectedNodeType, data: { ...n.data, label: selectedNodeType, config: parsed } } : n));
    alert('Узел обновлён');
  };
  const onDeleteSelectedNode = () => {
    if (!selectedNodeId) return;
    onRemoveNode(selectedNodeId);
    setSelectedNodeId(null);
  };

  return (
    <div className="lg:col-span-3">
      <div className="bg-white rounded-2xl shadow-lg p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-800">Pipeline Builder</h2>
          <div className="space-x-2">
            <button onClick={onSave} className="px-4 py-2 rounded-md font-medium bg-gray-100 hover:bg-gray-200">Сохранить</button>
            <button onClick={onRun} className="px-4 py-2 rounded-md font-medium bg-blue-600 text-white hover:bg-blue-700">Запустить</button>
            {taskId && <button onClick={async()=>{ try { await revokePipeline(taskId); } catch(e){ console.error(e);} }} className="px-4 py-2 rounded-md font-medium bg-red-600 text-white hover:bg-red-700">Остановить</button>}
            <button onClick={exportJson} className="px-4 py-2 rounded-md font-medium bg-gray-100 hover:bg-gray-200">Экспорт</button>
            <label className="px-4 py-2 rounded-md font-medium bg-gray-100 hover:bg-gray-200 cursor-pointer">
              Импорт
              <input key={importInputKey} type="file" accept="application/json" className="hidden" onChange={onImportFile} />
            </label>
            <button onClick={()=>{ setShowBacktests(v=>!v); if(!showBacktests) refreshBacktests(); }} className="px-4 py-2 rounded-md font-medium bg-gray-100 hover:bg-gray-200">История</button>
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
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Сохранённый Pipeline ID</label>
            <div className="flex space-x-2">
              <input value={savedPipelineId} onChange={(e)=>setSavedPipelineId(e.target.value)} className="w-2/3 p-3 border border-gray-300 rounded-lg" placeholder="id после сохранения" />
              <input value={loadPipelineId} onChange={(e)=>setLoadPipelineId(e.target.value)} className="w-1/3 p-3 border border-gray-300 rounded-lg" placeholder="id для загрузки" />
            </div>
            <div className="mt-2">
              <button onClick={onLoad} className="px-3 py-2 rounded bg-indigo-600 text-white">Загрузить</button>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg space-y-4">
          <ReactFlowProvider>
            <DnDCanvas
              nodes={nodes}
              edges={edges}
              setNodes={setNodes}
              setEdges={setEdges}
              onConnect={onConnect}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              nodeTypes={NODE_TYPES}
            />
          </ReactFlowProvider>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <div>
                <div className="text-sm font-medium text-gray-800 mb-2">Добавить узел</div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                  <input value={newNodeId} onChange={(e)=>setNewNodeId(e.target.value)} className="p-3 border rounded" placeholder="id (например, ind1)" />
                  <select value={newNodeType} onChange={(e)=>setNewNodeType(e.target.value)} className="p-3 border rounded">
                    {NODE_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                  <input value={newNodeConfig} onChange={(e)=>setNewNodeConfig(e.target.value)} className="p-3 border rounded" placeholder='{"key":"value"}' />
                  <button onClick={onAddNode} className="px-4 py-2 rounded bg-emerald-600 text-white">Добавить узел</button>
                </div>
              </div>

              <div>
                <div className="text-sm font-medium text-gray-800 mb-2">Список узлов</div>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-xs border rounded">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="px-2 py-1 text-left">id</th>
                        <th className="px-2 py-1 text-left">type</th>
                        <th className="px-2 py-1 text-left">config</th>
                        <th className="px-2 py-1" />
                      </tr>
                    </thead>
                    <tbody>
                      {nodes.map(n => (
                        <tr key={n.id} className="odd:bg-white even:bg-gray-50">
                          <td className="px-2 py-1">{n.id}</td>
                          <td className="px-2 py-1">{n.type}</td>
                          <td className="px-2 py-1"><code>{JSON.stringify(n.data?.config || {})}</code></td>
                          <td className="px-2 py-1 text-right"><button onClick={()=>onRemoveNode(n.id)} className="px-2 py-1 text-red-600">Удалить</button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div>
                <div className="text-sm font-medium text-gray-800 mb-2">Добавить ребро</div>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                  <select value={newEdgeSource} onChange={(e)=>setNewEdgeSource(e.target.value)} className="p-3 border rounded">
                    {nodes.map(n => <option key={n.id} value={n.id}>{n.id}</option>)}
                  </select>
                  <select value={newEdgeTarget} onChange={(e)=>setNewEdgeTarget(e.target.value)} className="p-3 border rounded">
                    {nodes.map(n => <option key={n.id} value={n.id}>{n.id}</option>)}
                  </select>
                  <div />
                  <button onClick={onAddEdge} className="px-4 py-2 rounded bg-sky-600 text-white">Добавить ребро</button>
                </div>
              </div>

              <div>
                <div className="text-sm font-medium text-gray-800 mb-2">Рёбра</div>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-xs border rounded">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="px-2 py-1 text-left">id</th>
                        <th className="px-2 py-1 text-left">source</th>
                        <th className="px-2 py-1 text-left">target</th>
                        <th className="px-2 py-1" />
                      </tr>
                    </thead>
                    <tbody>
                      {edges.map(e => (
                        <tr key={e.id} className="odd:bg-white even:bg-gray-50">
                          <td className="px-2 py-1">{e.id}</td>
                          <td className="px-2 py-1">{e.source}</td>
                          <td className="px-2 py-1">{e.target}</td>
                          <td className="px-2 py-1 text-right"><button onClick={()=>onRemoveEdge(e.id)} className="px-2 py-1 text-red-600">Удалить</button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="bg-white rounded border p-4 space-y-3">
              <div className="text-sm font-medium text-gray-800">Параметры узла</div>
              {selectedNodeId ? (
                <>
                  <div className="text-xs text-gray-500">id: <span className="font-mono">{selectedNodeId}</span></div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Тип</label>
                    <select value={selectedNodeType} onChange={(e)=>setSelectedNodeType(e.target.value)} className="w-full p-2 border rounded">
                      {NODE_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Config (JSON)</label>
                    <textarea value={selectedNodeConfig} onChange={(e)=>setSelectedNodeConfig(e.target.value)} className="w-full h-48 p-2 border rounded font-mono text-xs"/>
                  </div>
                  {PRESET_VARIANTS[selectedNodeType] && (
                    <div>
                      <div className="block text-xs text-gray-600 mb-1">Пресеты</div>
                      <div className="flex flex-wrap gap-2">
                        {PRESET_VARIANTS[selectedNodeType].map(p => (
                          <button key={p.label}
                                  onClick={() => setSelectedNodeConfig(JSON.stringify(p.config, null, 2))}
                                  className="px-2 py-1 text-xs border rounded bg-white hover:bg-gray-50">
                            {p.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="flex gap-2">
                    <button onClick={onUpdateSelectedNode} className="px-3 py-2 bg-emerald-600 text-white rounded">Сохранить</button>
                    <button onClick={onDeleteSelectedNode} className="px-3 py-2 bg-red-600 text-white rounded">Удалить</button>
                  </div>
                </>
              ) : (
                <div className="text-xs text-gray-500">Выберите узел на графе, чтобы отредактировать параметры</div>
              )}
            </div>
          </div>
        </div>

        {showBacktests && (
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm font-medium text-gray-800">История бэктестов</div>
              <button onClick={refreshBacktests} className="px-2 py-1 text-xs border rounded bg-white hover:bg-gray-100">Обновить</button>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs border rounded">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="px-2 py-1 text-left">id</th>
                    <th className="px-2 py-1 text-left">created_at</th>
                    <th className="px-2 py-1 text-left">timeframe</th>
                    <th className="px-2 py-1 text-left">metrics</th>
                    <th className="px-2 py-1 text-left">artifacts</th>
                  </tr>
                </thead>
                <tbody>
                  {backtests.map(bt => (
                    <tr key={bt.id} className="odd:bg-white even:bg-gray-50">
                      <td className="px-2 py-1">{bt.id}</td>
                      <td className="px-2 py-1">{bt.created_at}</td>
                      <td className="px-2 py-1">{bt.timeframe}</td>
                      <td className="px-2 py-1"><code className="break-all">{bt.metrics_json ? JSON.stringify(bt.metrics_json) : ''}</code></td>
                      <td className="px-2 py-1"><code className="break-all">{bt.artifacts ? JSON.stringify(bt.artifacts) : ''}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {taskId && (
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-sm text-gray-600">Task: <span className="font-medium">{taskId}</span></div>
            <div className="mt-2">
              <div className="mb-2 text-sm">Состояние: <span className="font-medium">{task?.state || 'PENDING'}</span></div>
              <div className="w-full bg-gray-200 rounded h-2 overflow-hidden">
                <div className="bg-blue-600 h-2" style={{ width: `${Math.min(100, task?.meta?.progress || 0)}%` }} />
              </div>
              {task?.meta?.metrics && (
                <div className="mt-2 text-xs bg-white p-3 rounded border">
                  <div className="font-medium mb-1">Metrics</div>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(task.meta.metrics).map(([k,v]) => (
                      <div key={k} className="flex justify-between"><span className="text-gray-600">{k}</span><span className="font-mono">{String(v)}</span></div>
                    ))}
                  </div>
                  {task.meta.metrics.PnL !== undefined && (
                    <div className="mt-3 grid grid-cols-2 gap-2">
                      <div className="flex justify-between"><span className="text-gray-600">PnL</span><span className="font-mono">{task.meta.metrics.PnL}</span></div>
                      <div className="flex justify-between"><span className="text-gray-600">WinRate</span><span className="font-mono">{task.meta.metrics.WinRate}</span></div>
                      <div className="flex justify-between"><span className="text-gray-600">Sharpe</span><span className="font-mono">{task.meta.metrics.Sharpe}</span></div>
                      <div className="flex justify-between"><span className="text-gray-600">Sortino</span><span className="font-mono">{task.meta.metrics.Sortino}</span></div>
                      <div className="flex justify-between"><span className="text-gray-600">MaxDrawdown</span><span className="font-mono">{task.meta.metrics.MaxDrawdown}</span></div>
                    </div>
                  )}
                  {task.meta.metrics.artifacts && (task.meta.metrics.artifacts.equity_csv || task.meta.metrics.artifacts.trades_csv) && (
                    <div className="mt-3 p-2 rounded border bg-gray-50">
                      <div className="font-medium mb-1">Артефакты</div>
                      {task.meta.metrics.artifacts.equity_csv && (
                        <div className="flex items-center justify-between text-[11px] break-all">
                          <span className="text-gray-600 mr-2">equity_csv:</span>
                          <a href={`/pipeline/artifacts/${encodeURIComponent(task.meta.metrics.artifacts.equity_csv)}`} target="_blank" rel="noreferrer" className="font-mono text-blue-600 hover:underline">{task.meta.metrics.artifacts.equity_csv}</a>
                        </div>
                      )}
                      {task.meta.metrics.artifacts.trades_csv && (
                        <div className="flex items-center justify-between text-[11px] break-all mt-1">
                          <span className="text-gray-600 mr-2">trades_csv:</span>
                          <a href={`/pipeline/artifacts/${encodeURIComponent(task.meta.metrics.artifacts.trades_csv)}`} target="_blank" rel="noreferrer" className="font-mono text-blue-600 hover:underline">{task.meta.metrics.artifacts.trades_csv}</a>
                        </div>
                      )}
                      <div className="mt-2 flex gap-2">
                        <button onClick={() => copyToClipboard(task.meta.metrics.artifacts.equity_csv)} className="px-2 py-1 border rounded bg-white hover:bg-gray-100 text-xs">Скопировать путь</button>
                        {task.meta.metrics.artifacts.trades_csv && (
                          <button onClick={() => copyToClipboard(task.meta.metrics.artifacts.trades_csv)} className="px-2 py-1 border rounded bg-white hover:bg-gray-100 text-xs">Скопировать trades</button>
                        )}
                      </div>
                    </div>
                  )}
                  {task.meta.metrics.per_coin_pnl && (
                    <div className="mt-3 p-2 rounded border bg-gray-50">
                      <div className="font-medium mb-1">PnL по монетам</div>
                      <div className="grid grid-cols-2 gap-2 text-[11px]">
                        {Object.entries(task.meta.metrics.per_coin_pnl).map(([coin, val]) => (
                          <div key={coin} className="flex justify-between"><span className="text-gray-600">{coin}</span><span className="font-mono">{val}</span></div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
              {task?.meta && (
                <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-auto max-h-64">{JSON.stringify(task.meta, null, 2)}</pre>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


function DnDCanvas({ nodes, edges, setNodes, setEdges, onConnect, onNodesChange, onEdgesChange, onNodeClick, nodeTypes }) {
  const { screenToFlowPosition } = useReactFlow();
  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);
  const onDrop = useCallback((event) => {
    event.preventDefault();
    const raw = event.dataTransfer.getData('application/reactflow');
    if (!raw) return;
    const data = JSON.parse(raw);
    const type = data?.type || 'DataSource';
    const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
    const id = `${type.toLowerCase()}_${nodes.length + 1}`;
    const cfg = DEFAULT_CONFIGS[type] || {};
    const newNode = { id, type, position, data: { label: type, config: cfg } };
    setNodes((nds) => nds.concat(newNode));
  }, [DEFAULT_CONFIGS, nodes.length, screenToFlowPosition, setNodes]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-6 gap-4">
      <div className="lg:col-span-1">
        <div className="text-sm font-medium text-gray-800 mb-2">Палитра</div>
        <div className="grid grid-cols-2 lg:grid-cols-1 gap-2">
          {nodeTypes.map((t) => (
            <div key={t}
                 draggable
                 onDragStart={(e)=>{
                   e.dataTransfer.setData('application/reactflow', JSON.stringify({ type: t }));
                   e.dataTransfer.effectAllowed = 'move';
                 }}
                 className="cursor-move px-3 py-2 bg-white border rounded text-xs text-gray-700 hover:bg-gray-100">
              {t}
            </div>
          ))}
        </div>
      </div>
      <div className="lg:col-span-5">
        <div className="h-[400px] w-full rounded border overflow-hidden" onDrop={onDrop} onDragOver={onDragOver}>
          <ReactFlow nodes={nodes} edges={edges} onConnect={onConnect} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onNodeClick={onNodeClick} fitView>
            <MiniMap />
            <Controls />
            <Background />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}


