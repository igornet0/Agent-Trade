import React, { useState } from 'react';

const TradeTimeTrainPanel = ({ config, onConfigChange }) => {
    const [localConfig, setLocalConfig] = useState({
        model_type: 'lightgbm',
        n_estimators: 100,
        learning_rate: 0.1,
        max_depth: 6,
        num_leaves: 31,
        depth: 6,
        threshold: 0.02,
        technical_indicators: ['sma', 'rsi', 'macd', 'bb'],
        news_integration: true,
        feature_scaling: true,
        val_split: 0.2,
        test_split: 0.2,
        ...config
    });

    const handleChange = (field, value) => {
        const newConfig = { ...localConfig, [field]: value };
        setLocalConfig(newConfig);
        onConfigChange(newConfig);
    };

    const handleCheckboxChange = (field, checked) => {
        handleChange(field, checked);
    };

    const handleArrayChange = (field, value, checked) => {
        let newArray = [...localConfig[field]];
        if (checked) {
            if (!newArray.includes(value)) {
                newArray.push(value);
            }
        } else {
            newArray = newArray.filter(item => item !== value);
        }
        handleChange(field, newArray);
    };

    return (
        <div className="space-y-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-900">Trade_time Configuration</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Model Type */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Model Type
                    </label>
                    <select
                        value={localConfig.model_type}
                        onChange={(e) => handleChange('model_type', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <option value="lightgbm">LightGBM</option>
                        <option value="catboost">CatBoost</option>
                        <option value="random_forest">Random Forest</option>
                    </select>
                </div>

                {/* N Estimators */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        N Estimators
                    </label>
                    <input
                        type="number"
                        min="50"
                        max="1000"
                        value={localConfig.n_estimators}
                        onChange={(e) => handleChange('n_estimators', parseInt(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                {/* Learning Rate */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Learning Rate
                    </label>
                    <input
                        type="number"
                        step="0.01"
                        min="0.01"
                        max="0.3"
                        value={localConfig.learning_rate}
                        onChange={(e) => handleChange('learning_rate', parseFloat(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                {/* Max Depth */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Max Depth
                    </label>
                    <input
                        type="number"
                        min="3"
                        max="15"
                        value={localConfig.max_depth}
                        onChange={(e) => handleChange('max_depth', parseInt(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                {/* Num Leaves (LightGBM specific) */}
                {localConfig.model_type === 'lightgbm' && (
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Num Leaves
                        </label>
                        <input
                            type="number"
                            min="10"
                            max="100"
                            value={localConfig.num_leaves}
                            onChange={(e) => handleChange('num_leaves', parseInt(e.target.value))}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                )}

                {/* Depth (CatBoost specific) */}
                {localConfig.model_type === 'catboost' && (
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Depth
                        </label>
                        <input
                            type="number"
                            min="3"
                            max="15"
                            value={localConfig.depth}
                            onChange={(e) => handleChange('depth', parseInt(e.target.value))}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                )}

                {/* Threshold */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Signal Threshold
                    </label>
                    <input
                        type="number"
                        step="0.01"
                        min="0.01"
                        max="0.1"
                        value={localConfig.threshold}
                        onChange={(e) => handleChange('threshold', parseFloat(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                {/* Validation Split */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Validation Split
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        min="0.1"
                        max="0.4"
                        value={localConfig.val_split}
                        onChange={(e) => handleChange('val_split', parseFloat(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                {/* Test Split */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Test Split
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        min="0.1"
                        max="0.4"
                        value={localConfig.test_split}
                        onChange={(e) => handleChange('test_split', parseFloat(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                </div>
            </div>

            {/* Technical Indicators */}
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Technical Indicators
                </label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {['sma', 'rsi', 'macd', 'bb', 'atr', 'volume'].map((indicator) => (
                        <label key={indicator} className="flex items-center space-x-2">
                            <input
                                type="checkbox"
                                checked={localConfig.technical_indicators.includes(indicator)}
                                onChange={(e) => handleArrayChange('technical_indicators', indicator, e.target.checked)}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <span className="text-sm text-gray-700 capitalize">{indicator}</span>
                        </label>
                    ))}
                </div>
            </div>

            {/* Boolean Options */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <label className="flex items-center space-x-2">
                    <input
                        type="checkbox"
                        checked={localConfig.news_integration}
                        onChange={(e) => handleCheckboxChange('news_integration', e.target.checked)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">News Integration</span>
                </label>

                <label className="flex items-center space-x-2">
                    <input
                        type="checkbox"
                        checked={localConfig.feature_scaling}
                        onChange={(e) => handleCheckboxChange('feature_scaling', e.target.checked)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">Feature Scaling</span>
                </label>
            </div>

            {/* Configuration Summary */}
            <div className="bg-blue-50 p-3 rounded-md">
                <h4 className="text-sm font-medium text-blue-900 mb-2">Configuration Summary</h4>
                <div className="text-xs text-blue-800 space-y-1">
                    <div>Model: {localConfig.model_type.toUpperCase()}</div>
                    <div>Estimators: {localConfig.n_estimators}</div>
                    <div>Learning Rate: {localConfig.learning_rate}</div>
                    <div>Max Depth: {localConfig.max_depth}</div>
                    <div>Threshold: {localConfig.threshold}</div>
                    <div>Indicators: {localConfig.technical_indicators.join(', ')}</div>
                    <div>Data Splits: Train {((1 - localConfig.val_split - localConfig.test_split) * 100).toFixed(0)}% | Val {(localConfig.val_split * 100).toFixed(0)}% | Test {(localConfig.test_split * 100).toFixed(0)}%</div>
                </div>
            </div>
        </div>
    );
};

export default TradeTimeTrainPanel;
