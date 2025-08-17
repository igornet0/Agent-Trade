import React, { useEffect, useMemo, useState } from 'react';
import { trainModel, getTaskStatus } from '../../services/mlService';
import { get_coins } from '../../services/strategyService';

const AGENT_TYPES = [
  { value: 'AgentNews', label: 'News' },
  { value: 'AgentPredTime', label: 'Pred_time' },
  { value: 'AgentTradeTime', label: 'Trade_time' },
  { value: 'AgentRisk', label: 'Risk' },
  { value: 'AgentTradeAggregator', label: 'Trade (Aggregator)' },
];

const TIMEFRAMES = ['5m','15m','30m','1h','4h','1d'];

export default function ModelTrainForm() {
  const [name, setName] = useState('');
  const [agentType, setAgentType] = useState('AgentPredTime');
  const [timeframe, setTimeframe] = useState('5m');
  const [coins, setCoins] = useState([]);
  const [selectedCoinIds, setSelectedCoinIds] = useState([]);

  // Common train params
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.001);
  const [weightDecay, setWeightDecay] = useState(0.0001);

  // Type-specific configs (minimal)
  const [predSeqLen, setPredSeqLen] = useState(96);
  const [predLen, setPredLen] = useState(12);
  const [useNewsBG, setUseNewsBG] = useState(true);

  const [clfType, setClfType] = useState('LightGBM');
  const [targetScheme, setTargetScheme] = useState('direction3');

  const [riskModel, setRiskModel] = useState('XGBoost');

  const [aggMode, setAggMode] = useState('rules');

  const [submitting, setSubmitting] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [task, setTask] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const cs = await get_coins();
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
        if (t.ready) clearInterval(timer);
      } catch (e) { console.error(e); }
    };
    timer = setInterval(tick, 2000);
    return () => clearInterval(timer);
  }, [taskId]);

  const toggleCoin = (id) => {
    setSelectedCoinIds((prev) => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const onSubmit = async () => {
    if (!name || !agentType) return;
    setSubmitting(true);
    try {
      const payload = {
        name,
        type: agentType,
        timeframe,
        coins: selectedCoinIds,
        features: [],
        train_data: { epochs, batch_size: batchSize, learning_rate: learningRate, weight_decay: weightDecay },
      };
      if (agentType === 'AgentPredTime') {
        payload.pred_time_config = { model_name: 'LSTM', seq_len: predSeqLen, pred_len: predLen, indicators: ['SMA','RSI','MACD'], use_news_background: useNewsBG };
      }
      if (agentType === 'AgentTradeTime') {
        payload.trade_time_config = { classifier: clfType, target_scheme: targetScheme, use_news_background: useNewsBG };
      }
      if (agentType === 'AgentRisk') {
        payload.risk_config = { model_name: riskModel, features: ['balance','pnl','leverage','signals'] };
      }
      if (agentType === 'AgentTradeAggregator') {
        payload.trade_aggregator_config = { mode: aggMode, weights: { pred_time: 0.4, trade_time: 0.4, news: 0.1, risk: 0.1 }, rl_enabled: false };
      }

      const res = await trainModel(agentType, payload);
      if (res && res.task_id) setTaskId(res.task_id);
      alert('Задача обучения запущена');
    } catch (e) {
      console.error(e);
      alert('Ошибка запуска обучения');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800">Обучение модели</h2>
        <button onClick={onSubmit} disabled={submitting || !name}
          className={`px-4 py-2 rounded-md font-medium ${(!name) ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
        >{submitting ? 'Запуск...' : 'Запустить обучение'}</button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Название</label>
          <input value={name} onChange={(e)=>setName(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg" placeholder="Имя модели" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Тип модуля</label>
          <select value={agentType} onChange={(e)=>setAgentType(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
            {AGENT_TYPES.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Таймфрейм</label>
          <select value={timeframe} onChange={(e)=>setTimeframe(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
            {TIMEFRAMES.map(tf => <option key={tf} value={tf}>{tf}</option>)}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Эпохи</label>
          <input type="number" value={epochs} onChange={(e)=>setEpochs(Number(e.target.value))} className="w-full p-3 border border-gray-300 rounded-lg" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Batch size</label>
          <input type="number" value={batchSize} onChange={(e)=>setBatchSize(Number(e.target.value))} className="w-full p-3 border border-gray-300 rounded-lg" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">LR</label>
          <input type="number" step="0.0001" value={learningRate} onChange={(e)=>setLearningRate(Number(e.target.value))} className="w-full p-3 border border-gray-300 rounded-lg" />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Weight decay</label>
          <input type="number" step="0.0001" value={weightDecay} onChange={(e)=>setWeightDecay(Number(e.target.value))} className="w-full p-3 border border-gray-300 rounded-lg" />
        </div>
      </div>

      {agentType === 'AgentPredTime' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-gray-50 p-4 rounded-lg">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">seq_len</label>
            <input type="number" value={predSeqLen} onChange={(e)=>setPredSeqLen(Number(e.target.value))} className="w-full p-3 border border-gray-300 rounded-lg" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">pred_len</label>
            <input type="number" value={predLen} onChange={(e)=>setPredLen(Number(e.target.value))} className="w-full p-3 border border-gray-300 rounded-lg" />
          </div>
          <div className="flex items-center mt-7">
            <input id="newsbg" type="checkbox" checked={useNewsBG} onChange={(e)=>setUseNewsBG(e.target.checked)} className="mr-2" />
            <label htmlFor="newsbg" className="text-sm text-gray-700">Use News Background</label>
          </div>
        </div>
      )}

      {agentType === 'AgentTradeTime' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-gray-50 p-4 rounded-lg">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Classifier</label>
            <select value={clfType} onChange={(e)=>setClfType(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
              {['LightGBM','CatBoost','Transformer'].map(x => <option key={x} value={x}>{x}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Target scheme</label>
            <select value={targetScheme} onChange={(e)=>setTargetScheme(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
              {['direction3','direction5'].map(x => <option key={x} value={x}>{x}</option>)}
            </select>
          </div>
          <div className="flex items-center mt-7">
            <input id="newsbg2" type="checkbox" checked={useNewsBG} onChange={(e)=>setUseNewsBG(e.target.checked)} className="mr-2" />
            <label htmlFor="newsbg2" className="text-sm text-gray-700">Use News Background</label>
          </div>
        </div>
      )}

      {agentType === 'AgentRisk' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-gray-50 p-4 rounded-lg">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Модель риска</label>
            <select value={riskModel} onChange={(e)=>setRiskModel(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
              {['XGBoost','Heuristic','RandomForest'].map(x => <option key={x} value={x}>{x}</option>)}
            </select>
          </div>
        </div>
      )}

      {agentType === 'AgentTradeAggregator' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-gray-50 p-4 rounded-lg">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Режим</label>
            <select value={aggMode} onChange={(e)=>setAggMode(e.target.value)} className="w-full p-3 border border-gray-300 rounded-lg">
              {['rules','rl'].map(x => <option key={x} value={x}>{x}</option>)}
            </select>
          </div>
        </div>
      )}

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
  );
}


