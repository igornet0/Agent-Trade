import React, { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, { Background, Controls, addEdge, MiniMap, ReactFlowProvider, useReactFlow } from 'reactflow';
import 'reactflow/dist/style.css';
import { savePipeline, runPipeline, loadPipeline, revokePipeline, getBacktests, getBacktest } from '../../services/pipelineService';
import { getTaskStatus } from '../../services/mlService';
import BacktestHistory from '../profile/BacktestHistory';

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

  // View mode state
  const [viewMode, setViewMode] = useState('builder'); // 'builder' or 'history'

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

  const onImportFile = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          if (data.nodes && data.edges) {
            setNodes(data.nodes.map(n => ({
              ...n,
              position: n.position || { x: Math.random() * 400, y: Math.random() * 400 },
              data: { label: n.type, config: n.config || {} }
            })));
            setEdges(data.edges);
            if (data.timeframe) setTimeframe(data.timeframe);
            if (data.start) setStart(data.start);
            if (data.end) setEnd(data.end);
            setImportInputKey(String(Date.now()));
          }
        } catch (error) {
          console.error('Error parsing import file:', error);
          alert('Ошибка при импорте файла');
        }
      };
      reader.readAsText(file);
    }
  }, []);

  const onLoad = useCallback(async () => {
    if (!loadPipelineId) return;
    try {
      const data = await loadPipeline(loadPipelineId);
      if (data.nodes && data.edges) {
        setNodes(data.nodes.map(n => ({
          ...n,
          position: n.position || { x: Math.random() * 400, y: Math.random() * 400 },
          data: { label: n.type, config: n.config || {} }
        })));
        setEdges(data.edges);
        if (data.timeframe) setTimeframe(data.timeframe);
        if (data.start) setStart(data.start);
        if (data.end) setEnd(data.end);
        setSavedPipelineId(loadPipelineId);
      }
    } catch (error) {
      console.error('Error loading pipeline:', error);
      alert('Ошибка при загрузке пайплайна');
    }
  }, [loadPipelineId]);

  const onSave = useCallback(async () => {
    try {
      const response = await savePipeline(pipelineConfig);
      setSavedPipelineId(response.id);
      alert(`Пайплайн сохранен с ID: ${response.id}`);
    } catch (error) {
      console.error('Error saving pipeline:', error);
      alert('Ошибка при сохранении пайплайна');
    }
  }, [pipelineConfig]);

  const onRun = useCallback(async () => {
    try {
      const response = await runPipeline(null, pipelineConfig);
      setTaskId(response.task_id);
      alert(`Пайплайн запущен, Task ID: ${response.task_id}`);
    } catch (error) {
      console.error('Error running pipeline:', error);
      alert('Ошибка при запуске пайплайна');
    }
  }, [pipelineConfig]);

  const exportJson = useCallback(() => {
    const dataStr = JSON.stringify(pipelineConfig, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'pipeline-config.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [pipelineConfig]);

  const onRemoveNode = (id) => {
    setNodes(prev => prev.filter(n => n.id !== id));
    setEdges(prev => prev.filter(e => e.source !== id && e.target !== id));
  };

  const onAddNode = useCallback(() => {
    if (!newNodeId) return alert('Укажите id узла');
    if (nodes.some(n => n.id === newNodeId)) return alert('Узел с таким id уже существует');
    let parsed = {};
    try { parsed = newNodeConfig ? JSON.parse(newNodeConfig) : {}; } catch { return alert('config должен быть валидным JSON'); }
    const baseCfg = Object.keys(parsed).length ? parsed : (DEFAULT_CONFIGS[newNodeType] || {});
    setNodes(prev => [...prev, { id: newNodeId, type: newNodeType, position: { x: 200, y: 200 + prev.length * 40 }, data: { label: newNodeType, config: baseCfg } }]);
  }, [newNodeId, newNodeType, newNodeConfig, nodes]);

  const onAddEdge = useCallback(() => {
    if (!newEdgeSource || !newEdgeTarget) return alert('Укажите source и target');
    if (newEdgeSource === newEdgeTarget) return alert('source и target не могут быть одинаковыми');
    const edgeId = `e_${newEdgeSource}_${newEdgeTarget}`;
    if (edges.some(e => e.id === edgeId)) return alert('Такое ребро уже существует');
    setEdges(prev => [...prev, { id: edgeId, source: newEdgeSource, target: newEdgeTarget }]);
  }, [newEdgeSource, newEdgeTarget, edges]);

  const onRemoveEdge = useCallback((id) => {
    setEdges(prev => prev.filter(e => e.id !== id));
  }, []);

  const onNodeClick = useCallback((event, node) => {
    setSelectedNodeId(node.id);
    setSelectedNodeType(node.type);
    setSelectedNodeConfig(JSON.stringify(node.data?.config || {}, null, 2));
  }, []);

  const onUpdateSelectedNode = useCallback(() => {
    if (!selectedNodeId) return;
    let parsed = {};
    try { parsed = JSON.parse(selectedNodeConfig); } catch { return alert('config должен быть валидным JSON'); }
    setNodes(prev => prev.map(n => n.id === selectedNodeId ? { ...n, type: selectedNodeType, data: { ...n.data, config: parsed } } : n));
  }, [selectedNodeId, selectedNodeType, selectedNodeConfig]);

  const onDeleteSelectedNode = useCallback(() => {
    if (!selectedNodeId) return;
    onRemoveNode(selectedNodeId);
    setSelectedNodeId(null);
    setSelectedNodeType('DataSource');
    setSelectedNodeConfig('{}');
  }, [selectedNodeId, onRemoveNode]);

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

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Pipeline Builder</h1>
          <p className="mt-2 text-gray-600">
            Создайте и настройте ML пайплайны для торговли
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="mb-6 bg-white rounded-lg shadow p-4">
          <div className="flex space-x-2">
            <button
              onClick={() => setViewMode('builder')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                viewMode === 'builder'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              🏗️ Pipeline Builder
            </button>
            <button
              onClick={() => setViewMode('history')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                viewMode === 'history'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              📊 История бэктестов
            </button>
          </div>
        </div>

        {/* Content based on view mode */}
        {viewMode === 'builder' ? (
          <div className="bg-white rounded-lg shadow">
            {/* Existing Pipeline Builder content */}
            <div className="p-6 border-b border-gray-200">
              <div className="flex flex-wrap gap-2 mb-4">
                <button onClick={onSave} className="px-4 py-2 rounded-md font-medium bg-gray-100 hover:bg-gray-200">Сохранить</button>
                <button onClick={onRun} className="px-4 py-2 rounded-md font-medium bg-blue-600 text-white hover:bg-blue-700">Запустить</button>
                {taskId && <button onClick={async()=>{ try { await revokePipeline(taskId); } catch(e){ console.error(e);} }} className="px-4 py-2 rounded-md font-medium bg-red-600 text-white hover:bg-red-700">Остановить</button>}
                <button onClick={exportJson} className="px-4 py-2 rounded-md font-medium bg-gray-100 hover:bg-gray-200">Экспорт</button>
                <label className="px-4 py-2 rounded-md font-medium bg-gray-100 hover:bg-gray-200 cursor-pointer">
                  Импорт
                  <input type="file" accept=".json" onChange={onImportFile} className="hidden" key={importInputKey} />
                </label>
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
            </div>
          </div>
        ) : (
          <BacktestHistory />
        )}

        {/* Task Progress Display */}
        {taskId && (
          <div className="mt-6 bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Прогресс выполнения</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Task ID:</span>
                <span className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">{taskId}</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Состояние:</span>
                <span className={`px-2 py-1 text-xs rounded-full font-medium ${
                  task?.state === 'SUCCESS' ? 'bg-green-100 text-green-800' :
                  task?.state === 'FAILURE' ? 'bg-red-100 text-red-800' :
                  task?.state === 'PROGRESS' ? 'bg-blue-100 text-blue-800' :
                  'bg-yellow-100 text-yellow-800'
                }`}>
                  {task?.state || 'PENDING'}
                </span>
              </div>
              
              {task?.meta?.progress !== undefined && (
                <div>
                  <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>Прогресс</span>
                    <span>{task.meta.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${task.meta.progress}%` }}
                    ></div>
                  </div>
                </div>
              )}
              
              {task?.meta?.current_step && (
                <div>
                  <span className="text-sm text-gray-600">Текущий шаг:</span>
                  <span className="ml-2 text-sm font-medium">{task.meta.current_step}</span>
                </div>
              )}
              
              {task?.ready && task?.meta?.result && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="text-sm font-medium text-green-800 mb-2">Результат выполнения:</div>
                  <div className="text-sm text-green-700">
                    {typeof task.meta.result === 'object' ? (
                      <pre className="whitespace-pre-wrap text-xs">
                        {JSON.stringify(task.meta.result, null, 2)}
                      </pre>
                    ) : (
                      String(task.meta.result)
                    )}
                  </div>
                </div>
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


