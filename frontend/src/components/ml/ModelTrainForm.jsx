import React, { useEffect, useMemo, useState } from 'react';
import { trainModel, getTaskStatus } from '../../services/mlService';
import { get_coins } from '../../services/strategyService';

const AGENT_TYPES = [
  { value: 'AgentNews', label: 'News Model', description: 'Анализ новостей и их влияния на монеты' },
  { value: 'AgentPredTime', label: 'Pred_time Model', description: 'Прогнозирование цены на основе временных рядов' },
  { value: 'AgentTradeTime', label: 'Trade_time Model', description: 'Генерация торговых сигналов' },
  { value: 'AgentRisk', label: 'Risk Model', description: 'Оценка рисков и управление позициями' },
  { value: 'AgentTradeAggregator', label: 'Trade Aggregator', description: 'Финальный агрегатор всех сигналов' },
];

const TIMEFRAMES = ['5m','15m','30m','1h','4h','1d'];

// Конфигурации по умолчанию для каждого типа агента
const DEFAULT_CONFIGS = {
  AgentNews: {
    model_name: 'finbert',
    window_hours: 24,
    decay_factor: 0.95,
    min_confidence: 0.7,
    use_sentiment: true,
    use_entities: true
  },
  AgentPredTime: {
    model_name: 'LSTM',
    seq_len: 96,
    pred_len: 12,
    hidden_size: 128,
    num_layers: 2,
    dropout: 0.2,
    use_news_background: true,
    indicators: ['SMA', 'RSI', 'MACD', 'Bollinger'],
    target: 'price_change'
  },
  AgentTradeTime: {
    classifier: 'LightGBM',
    target_scheme: 'direction3',
    use_news_background: true,
    use_pred_time: true,
    features: ['price', 'volume', 'indicators', 'news_bg'],
    threshold: 0.6
  },
  AgentRisk: {
    model_name: 'XGBoost',
    max_leverage: 3,
    risk_limit_pct: 2,
    max_drawdown_pct: 20,
    per_trade_risk_pct: 1,
    features: ['balance', 'pnl', 'leverage', 'signals', 'market_volatility']
  },
  AgentTradeAggregator: {
    mode: 'rules',
    weights: { pred_time: 0.4, trade_time: 0.4, news: 0.1, risk: 0.1 },
    rl_enabled: false,
    gamma: 0.99,
    learning_rate: 0.001,
    min_confidence: 0.7
  }
};

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
  const [validationSplit, setValidationSplit] = useState(0.2);

  // Type-specific configs
  const [config, setConfig] = useState(DEFAULT_CONFIGS.AgentPredTime);

  const [submitting, setSubmitting] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [task, setTask] = useState(null);

  // Обновляем конфигурацию при изменении типа агента
  useEffect(() => {
    setConfig(DEFAULT_CONFIGS[agentType]);
  }, [agentType]);

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

  const updateConfig = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
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
        train_data: { 
          epochs, 
          batch_size: batchSize, 
          learning_rate: learningRate, 
          weight_decay: weightDecay,
          validation_split: validationSplit
        },
        config: config
      };

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

  const renderNewsConfig = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-gray-50 p-4 rounded-lg">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Модель NLP</label>
        <select 
          value={config.model_name} 
          onChange={(e) => updateConfig('model_name', e.target.value)} 
          className="w-full p-3 border border-gray-300 rounded-lg"
        >
          <option value="finbert">FinBERT</option>
          <option value="bert">BERT</option>
          <option value="roberta">RoBERTa</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Окно анализа (часы)</label>
        <input 
          type="number" 
          value={config.window_hours} 
          onChange={(e) => updateConfig('window_hours', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Фактор затухания</label>
        <input 
          type="number" 
          step="0.01" 
          value={config.decay_factor} 
          onChange={(e) => updateConfig('decay_factor', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Мин. уверенность</label>
        <input 
          type="number" 
          step="0.1" 
          value={config.min_confidence} 
          onChange={(e) => updateConfig('min_confidence', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div className="flex items-center">
        <input 
          id="use_sentiment" 
          type="checkbox" 
          checked={config.use_sentiment} 
          onChange={(e) => updateConfig('use_sentiment', e.target.checked)} 
          className="mr-2" 
        />
        <label htmlFor="use_sentiment" className="text-sm text-gray-700">Использовать сентимент</label>
      </div>
      <div className="flex items-center">
        <input 
          id="use_entities" 
          type="checkbox" 
          checked={config.use_entities} 
          onChange={(e) => updateConfig('use_entities', e.target.checked)} 
          className="mr-2" 
        />
        <label htmlFor="use_entities" className="text-sm text-gray-700">Использовать сущности</label>
      </div>
    </div>
  );

  const renderPredTimeConfig = () => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-gray-50 p-4 rounded-lg">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Модель</label>
        <select 
          value={config.model_name} 
          onChange={(e) => updateConfig('model_name', e.target.value)} 
          className="w-full p-3 border border-gray-300 rounded-lg"
        >
          <option value="LSTM">LSTM</option>
          <option value="GRU">GRU</option>
          <option value="Transformer">Transformer</option>
          <option value="Informer">Informer</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Длина последовательности</label>
        <input 
          type="number" 
          value={config.seq_len} 
          onChange={(e) => updateConfig('seq_len', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Длина прогноза</label>
        <input 
          type="number" 
          value={config.pred_len} 
          onChange={(e) => updateConfig('pred_len', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Размер скрытого слоя</label>
        <input 
          type="number" 
          value={config.hidden_size} 
          onChange={(e) => updateConfig('hidden_size', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Количество слоев</label>
        <input 
          type="number" 
          value={config.num_layers} 
          onChange={(e) => updateConfig('num_layers', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Dropout</label>
        <input 
          type="number" 
          step="0.1" 
          value={config.dropout} 
          onChange={(e) => updateConfig('dropout', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div className="flex items-center">
        <input 
          id="use_news_bg" 
          type="checkbox" 
          checked={config.use_news_background} 
          onChange={(e) => updateConfig('use_news_background', e.target.checked)} 
          className="mr-2" 
        />
        <label htmlFor="use_news_bg" className="text-sm text-gray-700">Использовать новостной фон</label>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Целевая переменная</label>
        <select 
          value={config.target} 
          onChange={(e) => updateConfig('target', e.target.value)} 
          className="w-full p-3 border border-gray-300 rounded-lg"
        >
          <option value="price_change">Изменение цены</option>
          <option value="price_direction">Направление цены</option>
          <option value="volatility">Волатильность</option>
        </select>
      </div>
    </div>
  );

  const renderTradeTimeConfig = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-gray-50 p-4 rounded-lg">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Классификатор</label>
        <select 
          value={config.classifier} 
          onChange={(e) => updateConfig('classifier', e.target.value)} 
          className="w-full p-3 border border-gray-300 rounded-lg"
        >
          <option value="LightGBM">LightGBM</option>
          <option value="CatBoost">CatBoost</option>
          <option value="XGBoost">XGBoost</option>
          <option value="RandomForest">Random Forest</option>
          <option value="Transformer">Transformer</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Схема целевой переменной</label>
        <select 
          value={config.target_scheme} 
          onChange={(e) => updateConfig('target_scheme', e.target.value)} 
          className="w-full p-3 border border-gray-300 rounded-lg"
        >
          <option value="direction3">3 класса (Buy/Sell/Hold)</option>
          <option value="direction5">5 классов (Strong Buy/Buy/Hold/Sell/Strong Sell)</option>
          <option value="binary">Бинарный (Buy/Hold)</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Порог уверенности</label>
        <input 
          type="number" 
          step="0.1" 
          value={config.threshold} 
          onChange={(e) => updateConfig('threshold', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div className="flex items-center">
        <input 
          id="use_pred_time" 
          type="checkbox" 
          checked={config.use_pred_time} 
          onChange={(e) => updateConfig('use_pred_time', e.target.checked)} 
          className="mr-2" 
        />
        <label htmlFor="use_pred_time" className="text-sm text-gray-700">Использовать прогноз цены</label>
      </div>
      <div className="flex items-center">
        <input 
          id="use_news_bg_trade" 
          type="checkbox" 
          checked={config.use_news_background} 
          onChange={(e) => updateConfig('use_news_background', e.target.checked)} 
          className="mr-2" 
        />
        <label htmlFor="use_news_bg_trade" className="text-sm text-gray-700">Использовать новостной фон</label>
      </div>
    </div>
  );

  const renderRiskConfig = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-gray-50 p-4 rounded-lg">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Модель риска</label>
        <select 
          value={config.model_name} 
          onChange={(e) => updateConfig('model_name', e.target.value)} 
          className="w-full p-3 border border-gray-300 rounded-lg"
        >
          <option value="XGBoost">XGBoost</option>
          <option value="heuristic">Эвристическая</option>
          <option value="RandomForest">Random Forest</option>
          <option value="NeuralNetwork">Нейронная сеть</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Макс. плечо</label>
        <input 
          type="number" 
          step="0.1" 
          value={config.max_leverage} 
          onChange={(e) => updateConfig('max_leverage', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Лимит риска (%)</label>
        <input 
          type="number" 
          step="0.1" 
          value={config.risk_limit_pct} 
          onChange={(e) => updateConfig('risk_limit_pct', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Макс. просадка (%)</label>
        <input 
          type="number" 
          step="0.1" 
          value={config.max_drawdown_pct} 
          onChange={(e) => updateConfig('max_drawdown_pct', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Риск на сделку (%)</label>
        <input 
          type="number" 
          step="0.1" 
          value={config.per_trade_risk_pct} 
          onChange={(e) => updateConfig('per_trade_risk_pct', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
    </div>
  );

  const renderAggregatorConfig = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-gray-50 p-4 rounded-lg">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Режим агрегации</label>
        <select 
          value={config.mode} 
          onChange={(e) => updateConfig('mode', e.target.value)} 
          className="w-full p-3 border border-gray-300 rounded-lg"
        >
          <option value="rules">Правила</option>
          <option value="rl">Reinforcement Learning</option>
          <option value="ensemble">Ансамбль</option>
        </select>
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Мин. уверенность</label>
        <input 
          type="number" 
          step="0.1" 
          value={config.min_confidence} 
          onChange={(e) => updateConfig('min_confidence', Number(e.target.value))} 
          className="w-full p-3 border border-gray-300 rounded-lg" 
        />
      </div>
      <div className="flex items-center">
        <input 
          id="rl_enabled" 
          type="checkbox" 
          checked={config.rl_enabled} 
          onChange={(e) => updateConfig('rl_enabled', e.target.checked)} 
          className="mr-2" 
        />
        <label htmlFor="rl_enabled" className="text-sm text-gray-700">Включить RL</label>
      </div>
      {config.rl_enabled && (
        <>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Gamma (дисконт)</label>
            <input 
              type="number" 
              step="0.01" 
              value={config.gamma} 
              onChange={(e) => updateConfig('gamma', Number(e.target.value))} 
              className="w-full p-3 border border-gray-300 rounded-lg" 
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Learning Rate</label>
            <input 
              type="number" 
              step="0.0001" 
              value={config.learning_rate} 
              onChange={(e) => updateConfig('learning_rate', Number(e.target.value))} 
              className="w-full p-3 border border-gray-300 rounded-lg" 
            />
          </div>
        </>
      )}
    </div>
  );

  const renderConfigSection = () => {
    switch (agentType) {
      case 'AgentNews':
        return renderNewsConfig();
      case 'AgentPredTime':
        return renderPredTimeConfig();
      case 'AgentTradeTime':
        return renderTradeTimeConfig();
      case 'AgentRisk':
        return renderRiskConfig();
      case 'AgentTradeAggregator':
        return renderAggregatorConfig();
      default:
        return null;
    }
  };

  const selectedAgentType = AGENT_TYPES.find(t => t.value === agentType);

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Обучение модели</h2>
          {selectedAgentType && (
            <p className="text-sm text-gray-600 mt-1">{selectedAgentType.description}</p>
          )}
        </div>
        <button 
          onClick={onSubmit} 
          disabled={submitting || !name}
          className={`px-4 py-2 rounded-md font-medium ${(!name) ? 'bg-gray-300 cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
        >
          {submitting ? 'Запуск...' : 'Запустить обучение'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Название модели</label>
          <input 
            value={name} 
            onChange={(e)=>setName(e.target.value)} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
            placeholder="Введите название модели" 
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Тип модуля</label>
          <select 
            value={agentType} 
            onChange={(e)=>setAgentType(e.target.value)} 
            className="w-full p-3 border border-gray-300 rounded-lg"
          >
            {AGENT_TYPES.map(t => (
              <option key={t.value} value={t.value}>{t.label}</option>
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

      <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Эпохи</label>
          <input 
            type="number" 
            value={epochs} 
            onChange={(e)=>setEpochs(Number(e.target.value))} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Batch size</label>
          <input 
            type="number" 
            value={batchSize} 
            onChange={(e)=>setBatchSize(Number(e.target.value))} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Learning Rate</label>
          <input 
            type="number" 
            step="0.0001" 
            value={learningRate} 
            onChange={(e)=>setLearningRate(Number(e.target.value))} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Weight Decay</label>
          <input 
            type="number" 
            step="0.0001" 
            value={weightDecay} 
            onChange={(e)=>setWeightDecay(Number(e.target.value))} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Validation Split</label>
          <input 
            type="number" 
            step="0.1" 
            value={validationSplit} 
            onChange={(e)=>setValidationSplit(Number(e.target.value))} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
          />
        </div>
      </div>

      {/* Специфичные настройки для каждого типа агента */}
      {renderConfigSection()}

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Монеты для обучения</label>
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
    </div>
  );
}


