import React, { useState, useEffect } from 'react';
import { mlService } from '../../services/mlService';

const TEST_METRICS = {
  'AgentNews': ['accuracy', 'precision', 'recall', 'f1_score', 'sentiment_accuracy'],
  'AgentPredTime': ['mae', 'mse', 'rmse', 'mape', 'directional_accuracy'],
  'AgentTradeTime': ['accuracy', 'precision', 'recall', 'f1_score', 'profit_factor', 'win_rate'],
  'AgentRisk': ['risk_accuracy', 'max_drawdown', 'sharpe_ratio', 'calmar_ratio'],
  'AgentTradeAggregator': ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
};

const ModelTester = () => {
  const [selectedModel, setSelectedModel] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedCoins, setSelectedCoins] = useState([]);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [timeframe, setTimeframe] = useState('5m');
  const [testResults, setTestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (taskId) {
      const interval = setInterval(() => {
        checkTaskStatus();
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [taskId]);

  const loadModels = async () => {
    try {
      const response = await mlService.getModels();
      setModels(response.models || []);
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const checkTaskStatus = async () => {
    if (!taskId) return;
    
    try {
      const status = await mlService.getTaskStatus(taskId);
      setTaskStatus(status);
      
      if (status.state === 'SUCCESS') {
        setTestResults(status.meta);
        setLoading(false);
        setTaskId(null);
      } else if (status.state === 'FAILURE') {
        console.error('Task failed:', status.meta);
        setLoading(false);
        setTaskId(null);
      }
    } catch (error) {
      console.error('Error checking task status:', error);
    }
  };

  const onTest = async () => {
    if (!selectedModel || selectedCoins.length === 0) {
      alert('Please select a model and at least one coin');
      return;
    }

    setLoading(true);
    setTestResults(null);

    try {
      const payload = {
        model_id: selectedModel.id,
        coins: selectedCoins,
        timeframe: timeframe,
        start_date: startDate,
        end_date: endDate,
        metrics: TEST_METRICS[selectedModel.type] || []
      };

      const response = await mlService.testModel(payload);
      setTaskId(response.task_id);
    } catch (error) {
      console.error('Error starting test:', error);
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'SUCCESS': return 'text-green-600';
      case 'FAILURE': return 'text-red-600';
      case 'PROGRESS': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  const formatMetric = (key, value) => {
    if (typeof value === 'number') {
      if (key.includes('accuracy') || key.includes('precision') || key.includes('recall') || key.includes('rate')) {
        return `${(value * 100).toFixed(2)}%`;
      }
      if (key.includes('ratio') || key.includes('factor')) {
        return value.toFixed(3);
      }
      return value.toFixed(4);
    }
    return value;
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Model Testing</h3>
        
        {/* Model Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Model
          </label>
          <select
            className="w-full p-2 border border-gray-300 rounded-md"
            value={selectedModel?.id || ''}
            onChange={(e) => {
              const model = models.find(m => m.id === parseInt(e.target.value));
              setSelectedModel(model);
            }}
          >
            <option value="">Choose a model...</option>
            {models.map(model => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.type}) - {model.status}
              </option>
            ))}
          </select>
        </div>

        {selectedModel && (
          <div className="mb-4 p-4 bg-gray-50 rounded-md">
            <h4 className="font-medium mb-2">Model Information</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div><span className="font-medium">Type:</span> {selectedModel.type}</div>
              <div><span className="font-medium">Timeframe:</span> {selectedModel.timeframe}</div>
              <div><span className="font-medium">Status:</span> {selectedModel.status}</div>
              <div><span className="font-medium">Version:</span> {selectedModel.version}</div>
            </div>
          </div>
        )}

        {/* Test Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Coins (comma-separated IDs)
            </label>
            <input
              type="text"
              className="w-full p-2 border border-gray-300 rounded-md"
              placeholder="1,2,3"
              value={selectedCoins.join(',')}
              onChange={(e) => setSelectedCoins(e.target.value.split(',').map(id => parseInt(id.trim())).filter(id => !isNaN(id)))}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Timeframe
            </label>
            <select
              className="w-full p-2 border border-gray-300 rounded-md"
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <option value="5m">5 minutes</option>
              <option value="15m">15 minutes</option>
              <option value="30m">30 minutes</option>
              <option value="1h">1 hour</option>
              <option value="4h">4 hours</option>
              <option value="1d">1 day</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Start Date
            </label>
            <input
              type="date"
              className="w-full p-2 border border-gray-300 rounded-md"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              End Date
            </label>
            <input
              type="date"
              className="w-full p-2 border border-gray-300 rounded-md"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
        </div>

        {/* Test Button */}
        <button
          onClick={onTest}
          disabled={loading || !selectedModel || selectedCoins.length === 0}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {loading ? 'Testing...' : 'Start Test'}
        </button>

        {/* Task Status */}
        {taskStatus && (
          <div className="mt-4 p-4 bg-blue-50 rounded-md">
            <div className="flex items-center justify-between">
              <span className="font-medium">Task Status:</span>
              <span className={`font-medium ${getStatusColor(taskStatus.state)}`}>
                {taskStatus.state}
              </span>
            </div>
            {taskStatus.status && (
              <p className="text-sm text-gray-600 mt-1">{taskStatus.status}</p>
            )}
            {taskStatus.current && taskStatus.total && (
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(taskStatus.current / taskStatus.total) * 100}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  {taskStatus.current} / {taskStatus.total}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Test Results */}
      {testResults && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Test Results</h3>
          
          {/* Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {Object.entries(testResults.test_results || {}).map(([key, value]) => {
              if (typeof value === 'number' && !key.includes('samples') && !key.includes('trades')) {
                return (
                  <div key={key} className="bg-gray-50 p-3 rounded-md">
                    <div className="text-sm font-medium text-gray-600 capitalize">
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div className="text-lg font-semibold">
                      {formatMetric(key, value)}
                    </div>
                  </div>
                );
              }
              return null;
            })}
          </div>

          {/* Recommendations */}
          {testResults.test_results?.recommendations && (
            <div className="mb-6">
              <h4 className="font-medium mb-2">Recommendations</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                {testResults.test_results.recommendations.map((rec, index) => (
                  <li key={index}>{rec}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Charts Placeholder */}
          <div className="mb-6">
            <h4 className="font-medium mb-2">Performance Charts</h4>
            <div className="bg-gray-100 p-8 rounded-md text-center text-gray-500">
              Charts will be displayed here
            </div>
          </div>

          {/* Confusion Matrix Placeholder */}
          {testResults.test_results?.confusion_matrix && (
            <div className="mb-6">
              <h4 className="font-medium mb-2">Confusion Matrix</h4>
              <div className="bg-gray-100 p-8 rounded-md text-center text-gray-500">
                Confusion matrix will be displayed here
              </div>
            </div>
          )}

          {/* Raw Results */}
          <details className="mt-6">
            <summary className="cursor-pointer font-medium text-gray-700">
              Raw Test Results
            </summary>
            <pre className="mt-2 p-4 bg-gray-100 rounded-md text-xs overflow-auto">
              {JSON.stringify(testResults, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
};

export default ModelTester;
