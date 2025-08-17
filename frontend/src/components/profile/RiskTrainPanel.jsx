import React, { useState } from 'react';

const RiskTrainPanel = ({ config, onConfigChange }) => {
    const [localConfig, setLocalConfig] = useState({
        model_type: 'xgboost',
        n_estimators: 100,
        learning_rate: 0.1,
        max_depth: 6,
        risk_weight: 0.6,
        volume_weight: 0.4,
        technical_indicators: ['sma', 'rsi', 'macd', 'bb', 'atr'],
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

    const handleWeightChange = (field, value) => {
        const newValue = parseFloat(value);
        const otherField = field === 'risk_weight' ? 'volume_weight' : 'risk_weight';
        const otherValue = localConfig[otherField];
        
        // Ensure weights sum to 1.0
        if (newValue + otherValue <= 1.0) {
            handleChange(field, newValue);
        }
    };

    return (
        <div className="space-y-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-900">Risk Configuration</h3>
            
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
                        <option value="xgboost">XGBoost</option>
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

                {/* Risk Weight */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Risk Weight
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        min="0.1"
                        max="0.9"
                        value={localConfig.risk_weight}
                        onChange={(e) => handleWeightChange('risk_weight', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                        Current: {localConfig.risk_weight} | Available: {(1 - localConfig.volume_weight).toFixed(1)}
                    </p>
                </div>

                {/* Volume Weight */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Volume Weight
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        min="0.1"
                        max="0.9"
                        value={localConfig.volume_weight}
                        onChange={(e) => handleWeightChange('volume_weight', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                        Current: {localConfig.volume_weight} | Available: {(1 - localConfig.risk_weight).toFixed(1)}
                    </p>
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
                    {['sma', 'rsi', 'macd', 'bb', 'atr', 'volume', 'price_range'].map((indicator) => (
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

            {/* Risk Assessment Info */}
            <div className="bg-yellow-50 p-3 rounded-md">
                <h4 className="text-sm font-medium text-yellow-900 mb-2">Risk Assessment Features</h4>
                <div className="text-xs text-yellow-800 space-y-1">
                    <div>• RSI-based risk scoring (overbought/oversold conditions)</div>
                    <div>• Volatility-based risk (ATR, price range analysis)</div>
                    <div>• Volume-based risk assessment</div>
                    <div>• Trend-based risk (stable trends = lower risk)</div>
                    <div>• News sentiment integration for market sentiment risk</div>
                </div>
            </div>

            {/* Configuration Summary */}
            <div className="bg-blue-50 p-3 rounded-md">
                <h4 className="text-sm font-medium text-blue-900 mb-2">Configuration Summary</h4>
                <div className="text-xs text-blue-800 space-y-1">
                    <div>Model: {localConfig.model_type.toUpperCase()}</div>
                    <div>Estimators: {localConfig.n_estimators}</div>
                    <div>Learning Rate: {localConfig.learning_rate}</div>
                    <div>Max Depth: {localConfig.max_depth}</div>
                    <div>Weights: Risk {localConfig.risk_weight} | Volume {localConfig.volume_weight}</div>
                    <div>Indicators: {localConfig.technical_indicators.join(', ')}</div>
                    <div>Data Splits: Train {((1 - localConfig.val_split - localConfig.test_split) * 100).toFixed(0)}% | Val {(localConfig.val_split * 100).toFixed(0)}% | Test {(localConfig.test_split * 100).toFixed(0)}%</div>
                </div>
            </div>
        </div>
    );
};

export default RiskTrainPanel;
