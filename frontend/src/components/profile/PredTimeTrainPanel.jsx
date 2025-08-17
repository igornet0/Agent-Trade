import React from 'react';

const PredTimeTrainPanel = ({ config, onChange }) => {
  const handleChange = (field, value) => {
    onChange({ ...config, [field]: value });
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Длина последовательности
          </label>
          <input
            type="number"
            min="10"
            max="1000"
            value={config.seq_len || 60}
            onChange={(e) => handleChange('seq_len', parseInt(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="60"
          />
          <p className="text-xs text-gray-500 mt-1">Количество временных точек для входа</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Горизонт прогноза
          </label>
          <input
            type="number"
            min="1"
            max="100"
            value={config.pred_len || 12}
            onChange={(e) => handleChange('pred_len', parseInt(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="12"
          />
          <p className="text-xs text-gray-500 mt-1">На сколько шагов вперед предсказывать</p>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Тип модели
        </label>
        <select
          value={config.model_type || 'LSTM'}
          onChange={(e) => handleChange('model_type', e.target.value)}
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="LSTM">LSTM</option>
          <option value="GRU">GRU</option>
          <option value="Transformer">Transformer</option>
        </select>
      </div>

      {config.model_type === 'LSTM' || config.model_type === 'GRU' ? (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Размер скрытого слоя
            </label>
            <input
              type="number"
              min="16"
              max="512"
              value={config.hidden_size || 128}
              onChange={(e) => handleChange('hidden_size', parseInt(e.target.value))}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              placeholder="128"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Количество слоев
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={config.num_layers || 2}
              onChange={(e) => handleChange('num_layers', parseInt(e.target.value))}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              placeholder="2"
            />
          </div>
        </div>
      ) : null}

      {config.model_type === 'Transformer' ? (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Размерность модели
            </label>
            <input
              type="number"
              min="64"
              max="1024"
              value={config.d_model || 256}
              onChange={(e) => handleChange('d_model', parseInt(e.target.value))}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              placeholder="256"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Количество голов внимания
            </label>
            <input
              type="number"
              min="1"
              max="32"
              value={config.n_heads || 8}
              onChange={(e) => handleChange('n_heads', parseInt(e.target.value))}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              placeholder="8"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Количество слоев
            </label>
            <input
              type="number"
              min="1"
              max="20"
              value={config.n_layers || 6}
              onChange={(e) => handleChange('n_layers', parseInt(e.target.value))}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              placeholder="6"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Размер FF сети
            </label>
            <input
              type="number"
              min="256"
              max="4096"
              value={config.d_ff || 1024}
              onChange={(e) => handleChange('d_ff', parseInt(e.target.value))}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              placeholder="1024"
            />
          </div>
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Dropout
          </label>
          <input
            type="number"
            min="0.0"
            max="0.5"
            step="0.05"
            value={config.dropout || 0.2}
            onChange={(e) => handleChange('dropout', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="0.2"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Размер батча
          </label>
          <input
            type="number"
            min="1"
            max="512"
            value={config.batch_size || 32}
            onChange={(e) => handleChange('batch_size', parseInt(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="32"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Темп обучения
          </label>
          <input
            type="number"
            min="0.000001"
            max="0.1"
            step="0.000001"
            value={config.learning_rate || 0.001}
            onChange={(e) => handleChange('learning_rate', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="0.001"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Количество эпох
          </label>
          <input
            type="number"
            min="1"
            max="1000"
            value={config.epochs || 100}
            onChange={(e) => handleChange('epochs', parseInt(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="100"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Терпение (early stopping)
        </label>
        <input
          type="number"
          min="1"
          max="100"
          value={config.patience || 20}
          onChange={(e) => handleChange('patience', parseInt(e.target.value))}
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          placeholder="20"
        />
        <p className="text-xs text-gray-500 mt-1">Количество эпох без улучшения для остановки</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Технические индикаторы
        </label>
        <div className="grid grid-cols-2 gap-2">
          {['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'Stoch', 'Williams_R'].map(indicator => (
            <label key={indicator} className="flex items-center">
              <input
                type="checkbox"
                checked={config.technical_indicators?.includes(indicator) || false}
                onChange={(e) => {
                  const indicators = config.technical_indicators || [];
                  if (e.target.checked) {
                    handleChange('technical_indicators', [...indicators, indicator]);
                  } else {
                    handleChange('technical_indicators', indicators.filter(i => i !== indicator));
                  }
                }}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm text-gray-700">{indicator}</span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Интеграция новостей
        </label>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={config.news_integration !== false}
            onChange={(e) => handleChange('news_integration', e.target.checked)}
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <span className="ml-2 text-sm text-gray-700">Использовать новостной фон как дополнительную фичу</span>
        </label>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Масштабирование фич
        </label>
        <select
          value={config.feature_scaling || 'standard'}
          onChange={(e) => handleChange('feature_scaling', e.target.value)}
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="none">Без масштабирования</option>
          <option value="standard">StandardScaler (Z-score)</option>
          <option value="minmax">MinMaxScaler (0-1)</option>
        </select>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Доля валидации
          </label>
          <input
            type="number"
            min="0.0"
            max="0.5"
            step="0.05"
            value={config.val_split || 0.2}
            onChange={(e) => handleChange('val_split', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="0.2"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Доля теста
          </label>
          <input
            type="number"
            min="0.0"
            max="0.5"
            step="0.05"
            value={config.test_split || 0.1}
            onChange={(e) => handleChange('test_split', parseFloat(e.target.value))}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="0.1"
          />
        </div>
      </div>
    </div>
  );
};

export default PredTimeTrainPanel;
