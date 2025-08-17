import React, { useState, useEffect } from 'react';

const TradeAggregatorTrainPanel = ({ config, onChange }) => {
  const [localConfig, setLocalConfig] = useState({
    mode: 'rules',
    weights: {
      pred_time: 0.4,
      trade_time: 0.4,
      risk: 0.2
    },
    thresholds: {
      buy_threshold: 0.6,
      sell_threshold: 0.4,
      hold_threshold: 0.3
    },
    risk_limits: {
      max_position_size: 0.1,
      max_leverage: 3.0,
      stop_loss_pct: 0.05,
      take_profit_pct: 0.15
    },
    portfolio: {
      max_coins: 10,
      rebalance_frequency: '1h',
      correlation_threshold: 0.7
    },
    // ML параметры
    n_estimators: 100,
    learning_rate: 0.1,
    max_depth: 6,
    technical_indicators: ['sma', 'rsi', 'macd', 'bb'],
    news_integration: true,
    feature_scaling: true,
    val_split: 0.2,
    test_split: 0.2
  });

  useEffect(() => {
    if (config) {
      setLocalConfig({ ...localConfig, ...config });
    }
  }, [config]);

  useEffect(() => {
    onChange(localConfig);
  }, [localConfig, onChange]);

  const updateConfig = (path, value) => {
    const newConfig = { ...localConfig };
    const keys = path.split('.');
    let current = newConfig;
    
    for (let i = 0; i < keys.length - 1; i++) {
      current = current[keys[i]];
    }
    current[keys[keys.length - 1]] = value;
    
    setLocalConfig(newConfig);
  };

  const updateWeights = (key, value) => {
    const newWeights = { ...localConfig.weights };
    newWeights[key] = parseFloat(value);
    
    // Нормализация весов
    const total = Object.values(newWeights).reduce((sum, w) => sum + w, 0);
    Object.keys(newWeights).forEach(k => {
      newWeights[k] = newWeights[k] / total;
    });
    
    updateConfig('weights', newWeights);
  };

  const updateThresholds = (key, value) => {
    const newThresholds = { ...localConfig.thresholds };
    newThresholds[key] = parseFloat(value);
    
    // Валидация порогов
    if (newThresholds.buy_threshold <= newThresholds.sell_threshold) {
      newThresholds.buy_threshold = newThresholds.sell_threshold + 0.1;
    }
    if (newThresholds.sell_threshold <= newThresholds.hold_threshold) {
      newThresholds.sell_threshold = newThresholds.hold_threshold + 0.1;
    }
    
    updateConfig('thresholds', newThresholds);
  };

  const updateRiskLimits = (key, value) => {
    const newRiskLimits = { ...localConfig.risk_limits };
    newRiskLimits[key] = parseFloat(value);
    updateConfig('risk_limits', newRiskLimits);
  };

  const updatePortfolio = (key, value) => {
    const newPortfolio = { ...localConfig.portfolio };
    newPortfolio[key] = value;
    updateConfig('portfolio', newPortfolio);
  };

  const toggleIndicator = (indicator) => {
    const newIndicators = [...localConfig.technical_indicators];
    const index = newIndicators.indexOf(indicator);
    
    if (index > -1) {
      newIndicators.splice(index, 1);
    } else {
      newIndicators.push(indicator);
    }
    
    updateConfig('technical_indicators', newIndicators);
  };

  return (
    <div className="space-y-6 p-4 bg-gray-50 rounded-lg">
      <h3 className="text-lg font-semibold text-gray-800">Trade Aggregator Configuration</h3>
      
      {/* Основные настройки */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Режим работы
          </label>
          <select
            value={localConfig.mode}
            onChange={(e) => updateConfig('mode', e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md"
          >
            <option value="rules">Rules-based</option>
            <option value="ml">ML-based</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Максимум монет в портфеле
          </label>
          <input
            type="number"
            min="1"
            max="50"
            value={localConfig.portfolio.max_coins}
            onChange={(e) => updatePortfolio('max_coins', parseInt(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md"
          />
        </div>
      </div>

      {/* Веса агрегации */}
      <div className="space-y-3">
        <h4 className="text-md font-medium text-gray-700">Веса агрегации сигналов</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Pred_time: {localConfig.weights.pred_time.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={localConfig.weights.pred_time}
              onChange={(e) => updateWeights('pred_time', e.target.value)}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Trade_time: {localConfig.weights.trade_time.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={localConfig.weights.trade_time}
              onChange={(e) => updateWeights('trade_time', e.target.value)}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Risk: {localConfig.weights.risk.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={localConfig.weights.risk}
              onChange={(e) => updateWeights('risk', e.target.value)}
              className="w-full"
            />
          </div>
        </div>
        
        <div className="text-xs text-gray-500">
          Сумма весов: {(localConfig.weights.pred_time + localConfig.weights.trade_time + localConfig.weights.risk).toFixed(2)}
        </div>
      </div>

      {/* Пороги решений */}
      <div className="space-y-3">
        <h4 className="text-md font-medium text-gray-700">Пороги принятия решений</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Buy threshold: {localConfig.thresholds.buy_threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.05"
              value={localConfig.thresholds.buy_threshold}
              onChange={(e) => updateThresholds('buy_threshold', e.target.value)}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Sell threshold: {localConfig.thresholds.sell_threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={localConfig.thresholds.sell_threshold}
              onChange={(e) => updateThresholds('sell_threshold', e.target.value)}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Hold threshold: {localConfig.thresholds.hold_threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={localConfig.thresholds.hold_threshold}
              onChange={(e) => updateThresholds('hold_threshold', e.target.value)}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Риск-лимиты */}
      <div className="space-y-3">
        <h4 className="text-md font-medium text-gray-700">Риск-менеджмент</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Макс. размер позиции: {(localConfig.risk_limits.max_position_size * 100).toFixed(1)}%
            </label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={localConfig.risk_limits.max_position_size}
              onChange={(e) => updateRiskLimits('max_position_size', e.target.value)}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Макс. плечо: {localConfig.risk_limits.max_leverage.toFixed(1)}x
            </label>
            <input
              type="range"
              min="1"
              max="10"
              step="0.5"
              value={localConfig.risk_limits.max_leverage}
              onChange={(e) => updateRiskLimits('max_leverage', e.target.value)}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Стоп-лосс: {(localConfig.risk_limits.stop_loss_pct * 100).toFixed(1)}%
            </label>
            <input
              type="range"
              min="0.01"
              max="0.2"
              step="0.01"
              value={localConfig.risk_limits.stop_loss_pct}
              onChange={(e) => updateRiskLimits('stop_loss_pct', e.target.value)}
              className="w-full"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-600 mb-1">
              Тейк-профит: {(localConfig.risk_limits.take_profit_pct * 100).toFixed(1)}%
            </label>
            <input
              type="range"
              min="0.05"
              max="0.5"
              step="0.01"
              value={localConfig.risk_limits.take_profit_pct}
              onChange={(e) => updateRiskLimits('take_profit_pct', e.target.value)}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Настройки портфеля */}
      <div className="space-y-3">
        <h4 className="text-md font-medium text-gray-700">Настройки портфеля</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Частота ребалансировки
            </label>
            <select
              value={localConfig.portfolio.rebalance_frequency}
              onChange={(e) => updatePortfolio('rebalance_frequency', e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md"
            >
              <option value="15m">15 минут</option>
              <option value="1h">1 час</option>
              <option value="4h">4 часа</option>
              <option value="1d">1 день</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Порог корреляции: {localConfig.portfolio.correlation_threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.3"
              max="0.9"
              step="0.05"
              value={localConfig.portfolio.correlation_threshold}
              onChange={(e) => updatePortfolio('correlation_threshold', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* ML параметры (если включен ML режим) */}
      {localConfig.mode === 'ml' && (
        <div className="space-y-3">
          <h4 className="text-md font-medium text-gray-700">ML параметры</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm text-gray-600 mb-1">
                Количество деревьев: {localConfig.n_estimators}
              </label>
              <input
                type="range"
                min="50"
                max="1000"
                step="50"
                value={localConfig.n_estimators}
                onChange={(e) => updateConfig('n_estimators', parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm text-gray-600 mb-1">
                Скорость обучения: {localConfig.learning_rate.toFixed(3)}
              </label>
              <input
                type="range"
                min="0.01"
                max="0.3"
                step="0.01"
                value={localConfig.learning_rate}
                onChange={(e) => updateConfig('learning_rate', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm text-gray-600 mb-1">
                Макс. глубина: {localConfig.max_depth}
              </label>
              <input
                type="range"
                min="3"
                max="15"
                step="1"
                value={localConfig.max_depth}
                onChange={(e) => updateConfig('max_depth', parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-600 mb-1">
                Доля валидации: {(localConfig.val_split * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.4"
                step="0.05"
                value={localConfig.val_split}
                onChange={(e) => updateConfig('val_split', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm text-gray-600 mb-1">
                Доля тестирования: {(localConfig.test_split * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.1"
                max="0.4"
                step="0.05"
                value={localConfig.test_split}
                onChange={(e) => updateConfig('test_split', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>
      )}

      {/* Технические индикаторы */}
      <div className="space-y-3">
        <h4 className="text-md font-medium text-gray-700">Технические индикаторы</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'stoch', 'cci'].map(indicator => (
            <label key={indicator} className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={localConfig.technical_indicators.includes(indicator)}
                onChange={() => toggleIndicator(indicator)}
                className="rounded border-gray-300"
              />
              <span className="text-sm text-gray-700">{indicator.toUpperCase()}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Дополнительные опции */}
      <div className="space-y-3">
        <h4 className="text-md font-medium text-gray-700">Дополнительные опции</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={localConfig.news_integration}
              onChange={(e) => updateConfig('news_integration', e.target.checked)}
              className="rounded border-gray-300"
            />
            <span className="text-sm text-gray-700">Интеграция новостей</span>
          </label>
          
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={localConfig.feature_scaling}
              onChange={(e) => updateConfig('feature_scaling', e.target.checked)}
              className="rounded border-gray-300"
            />
            <span className="text-sm text-gray-700">Масштабирование признаков</span>
          </label>
        </div>
      </div>

      {/* Сводка конфигурации */}
      <div className="bg-blue-50 p-4 rounded-lg">
        <h4 className="text-md font-medium text-blue-800 mb-2">Сводка конфигурации</h4>
        <div className="text-sm text-blue-700 space-y-1">
          <div>Режим: <strong>{localConfig.mode === 'rules' ? 'Rules-based' : 'ML-based'}</strong></div>
          <div>Веса: Pred_time({localConfig.weights.pred_time.toFixed(2)}) + Trade_time({localConfig.weights.trade_time.toFixed(2)}) + Risk({localConfig.weights.risk.toFixed(2)})</div>
                      <div>Пороги: Buy({localConfig.thresholds.buy_threshold.toFixed(2)}) &gt; Sell({localConfig.thresholds.sell_threshold.toFixed(2)}) &gt; Hold({localConfig.thresholds.hold_threshold.toFixed(2)})</div>
          <div>Риск: Позиция до {(localConfig.risk_limits.max_position_size * 100).toFixed(1)}%, плечо до {localConfig.risk_limits.max_leverage.toFixed(1)}x</div>
          <div>Портфель: Макс. {localConfig.portfolio.max_coins} монет, ребаланс {localConfig.portfolio.rebalance_frequency}</div>
          {localConfig.mode === 'ml' && (
            <div>ML: {localConfig.n_estimators} деревьев, lr={localConfig.learning_rate.toFixed(3)}, depth={localConfig.max_depth}</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TradeAggregatorTrainPanel;
