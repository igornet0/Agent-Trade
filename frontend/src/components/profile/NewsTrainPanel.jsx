import React from 'react';

const NewsTrainPanel = ({ config, onChange }) => {
  const handleChange = (field, value) => {
    onChange({ ...config, [field]: value });
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          NLP Модель
        </label>
        <select
          value={config.nlp_model || 'finbert'}
          onChange={(e) => handleChange('nlp_model', e.target.value)}
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="finbert">FinBERT (финансовый)</option>
          <option value="bert">BERT (общий)</option>
          <option value="distilbert">DistilBERT (быстрый)</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Окно влияния (часы)
        </label>
        <input
          type="number"
          min="1"
          max="168"
          value={config.window_hours || 24}
          onChange={(e) => handleChange('window_hours', parseInt(e.target.value))}
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          placeholder="24"
        />
        <p className="text-xs text-gray-500 mt-1">Сколько часов новость влияет на цену</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Коэффициент затухания
        </label>
        <input
          type="number"
          min="0.1"
          max="1.0"
          step="0.05"
          value={config.decay_factor || 0.95}
          onChange={(e) => handleChange('decay_factor', parseFloat(e.target.value))}
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          placeholder="0.95"
        />
        <p className="text-xs text-gray-500 mt-1">Скорость затухания влияния новости (0.95 = 5% затухание в час)</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Порог корреляции
        </label>
        <input
          type="number"
          min="0.0"
          max="1.0"
          step="0.05"
          value={config.correlation_threshold || 0.3}
          onChange={(e) => handleChange('correlation_threshold', parseFloat(e.target.value))}
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          placeholder="0.3"
        />
        <p className="text-xs text-gray-500 mt-1">Минимальная корреляция для оценки влияния</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Источники новостей
        </label>
        <div className="space-y-2">
          {['twitter', 'telegram', 'coindesk', 'cointelegraph', 'reddit'].map(source => (
            <label key={source} className="flex items-center">
              <input
                type="checkbox"
                checked={config.sources?.includes(source) || false}
                onChange={(e) => {
                  const sources = config.sources || [];
                  if (e.target.checked) {
                    handleChange('sources', [...sources, source]);
                  } else {
                    handleChange('sources', sources.filter(s => s !== source));
                  }
                }}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <span className="ml-2 text-sm text-gray-700 capitalize">{source}</span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Минимальное количество источников
        </label>
        <input
          type="number"
          min="1"
          max="10"
          value={config.min_sources || 2}
          onChange={(e) => handleChange('min_sources', parseInt(e.target.value))}
          className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          placeholder="2"
        />
        <p className="text-xs text-gray-500 mt-1">Минимум источников для расчета фона</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Принудительный пересчет
        </label>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={config.force_recalculate || false}
            onChange={(e) => handleChange('force_recalculate', e.target.checked)}
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <span className="ml-2 text-sm text-gray-700">Пересчитать даже если данные не изменились</span>
        </label>
      </div>
    </div>
  );
};

export default NewsTrainPanel;
