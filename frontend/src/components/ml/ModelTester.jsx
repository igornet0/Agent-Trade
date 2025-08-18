import React, { useState, useEffect } from 'react';
import { mlService } from '../../services/mlService';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  BarElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  TimeScale
} from 'chart.js';
import 'chartjs-adapter-date-fns';

ChartJS.register(LineElement, BarElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend, TimeScale);

const TEST_METRICS = {
  'AgentNews': ['accuracy', 'precision', 'recall', 'f1_score', 'sentiment_accuracy'],
  'AgentPredTime': ['mae', 'mse', 'rmse', 'mape', 'directional_accuracy'],
  'AgentTradeTime': ['accuracy', 'precision', 'recall', 'f1_score', 'profit_factor', 'win_rate'],
  'AgentRisk': ['risk_accuracy', 'max_drawdown', 'sharpe_ratio', 'calmar_ratio'],
  'AgentTradeAggregator': ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
};

const ITEMS_PER_PAGE = 10;

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

  // Model versions for Risk/TradeTime
  const [modelVersions, setModelVersions] = useState([]);
  const [selectedModelPath, setSelectedModelPath] = useState('');
  const [previewData, setPreviewData] = useState(null);

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [filterType, setFilterType] = useState('');
  const [filterStatus, setFilterStatus] = useState('');

  // Loading states
  const [loadingModels, setLoadingModels] = useState(false);
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);

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

  useEffect(() => {
    // Load available model versions for Risk/TradeTime when model changes
    const fetchModelVersions = async () => {
      setModelVersions([]);
      setSelectedModelPath('');
      setPreviewData(null);
      if (!selectedModel) return;
      
      setLoadingVersions(true);
      try {
        if (selectedModel.type === 'AgentRisk') {
          const res = await mlService.risk.getModels(selectedModel.id);
          setModelVersions(res.data?.models || res.models || []);
        } else if (selectedModel.type === 'AgentTradeTime') {
          const res = await mlService.tradeTime.getModels(selectedModel.id);
          setModelVersions(res.data?.models || res.models || []);
        }
      } catch (e) {
        console.error('Error loading model versions', e);
      } finally {
        setLoadingVersions(false);
      }
    };
    fetchModelVersions();
  }, [selectedModel]);

  const loadModels = async () => {
    setLoadingModels(true);
    try {
      const response = await mlService.listModels();
      setModels(response.models || []);
    } catch (error) {
      console.error('Error loading models:', error);
    } finally {
      setLoadingModels(false);
    }
  };

  const checkTaskStatus = async () => {
    if (!taskId) return;
    
    try {
      const status = await mlService.getTaskStatus(taskId);
      setTaskStatus(status);
      
      if (status.state === 'SUCCESS') {
        setTestResults(status.meta || status);
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

  const onPreviewSeries = async () => {
    if (!selectedModel || !selectedModelPath || selectedCoins.length === 0) {
      alert('Select model version and coin');
      return;
    }
    
    setLoadingPreview(true);
    try {
      if (selectedModel.type === 'AgentRisk') {
        const res = await mlService.risk.predict(selectedCoins[0], startDate, endDate, selectedModelPath);
        const data = res.data || res;
        setPreviewData({ type: 'risk', ...data });
      } else if (selectedModel.type === 'AgentTradeTime') {
        const res = await mlService.tradeTime.predict(selectedCoins[0], startDate, endDate, selectedModelPath);
        const data = res.data || res;
        setPreviewData({ type: 'trade_time', ...data });
      }
    } catch (e) {
      console.error('Error previewing series', e);
    } finally {
      setLoadingPreview(false);
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

  const getStatusBadge = (status) => {
    const colors = {
      'SUCCESS': 'bg-green-100 text-green-800',
      'FAILURE': 'bg-red-100 text-red-800',
      'PROGRESS': 'bg-blue-100 text-blue-800',
      'PENDING': 'bg-yellow-100 text-yellow-800'
    };
    return colors[status] || 'bg-gray-100 text-gray-800';
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

  const renderChart = (chart) => {
    if (!chart || !chart.type) return null;
    
    const commonOptions = {
      responsive: true,
      plugins: { 
        legend: { position: 'top' },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleColor: 'white',
          bodyColor: 'white',
          borderColor: 'rgba(255, 255, 255, 0.2)',
          borderWidth: 1
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      },
      scales: { 
        x: { display: true }, 
        y: { display: true } 
      }
    };

    if (chart.type === 'line') {
      const data = {
        labels: (chart.data.labels || chart.data.actual?.map((_, i) => i) || []),
        datasets: [
          chart.data.actual && {
            label: 'Actual',
            data: chart.data.actual,
            borderColor: 'rgba(99, 102, 241, 1)',
            backgroundColor: 'rgba(99, 102, 241, 0.2)',
            fill: false,
            tension: 0.2,
            pointRadius: 2,
            pointHoverRadius: 5
          },
          chart.data.predicted && {
            label: 'Predicted',
            data: chart.data.predicted,
            borderColor: 'rgba(16, 185, 129, 1)',
            backgroundColor: 'rgba(16, 185, 129, 0.2)',
            fill: false,
            tension: 0.2,
            pointRadius: 2,
            pointHoverRadius: 5
          }
        ].filter(Boolean)
      };
      return <Line data={data} options={commonOptions} />;
    }
    
    if (chart.type === 'histogram') {
      const data = {
        labels: chart.data.errors?.map((_, i) => i) || [],
        datasets: [
          {
            label: 'Errors',
            data: chart.data.errors || [],
            backgroundColor: 'rgba(239, 68, 68, 0.6)',
            borderColor: 'rgba(239, 68, 68, 1)',
            borderWidth: 1
          }
        ]
      };
      return <Bar data={data} options={commonOptions} />;
    }
    
    return null;
  };

  const renderPreview = () => {
    if (!previewData) return null;
    
    const commonOptions = { 
      responsive: true, 
      plugins: { 
        legend: { position: 'top' },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleColor: 'white',
          bodyColor: 'white'
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      },
      scales: { 
        x: { type: 'time' },
        y: { display: true } 
      }
    };

    if (previewData.type === 'risk') {
      const labels = (previewData.timestamp || []).map(ts => new Date(ts));
      const riskData = {
        labels,
        datasets: [
          {
            label: 'Risk score',
            data: previewData.risk_scores || [],
            borderColor: 'rgba(239, 68, 68, 1)',
            backgroundColor: 'rgba(239, 68, 68, 0.2)',
            tension: 0.2,
            pointRadius: 1,
            pointHoverRadius: 4
          },
          {
            label: 'Volume score',
            data: previewData.volume_scores || [],
            borderColor: 'rgba(14, 165, 233, 1)',
            backgroundColor: 'rgba(14, 165, 233, 0.2)',
            tension: 0.2,
            pointRadius: 1,
            pointHoverRadius: 4
          }
        ]
      };
      return (
        <div className="space-y-4">
          <h4 className="font-medium">Risk/Volume Series</h4>
          <Line data={riskData} options={commonOptions} />
        </div>
      );
    }
    
    if (previewData.type === 'trade_time') {
      const labels = (previewData.timestamp || []).map(ts => new Date(ts));
      const predSeries = {
        labels,
        datasets: [
          {
            label: 'Signal (Buy=1/Hold=0/Sell=-1)',
            data: (previewData.predictions || []).map(v => (typeof v === 'number' ? v : 0)),
            borderColor: 'rgba(34, 197, 94, 1)',
            backgroundColor: 'rgba(34, 197, 94, 0.2)',
            tension: 0.1,
            pointRadius: 1,
            pointHoverRadius: 4
          }
        ]
      };
      return (
        <div className="space-y-4">
          <h4 className="font-medium">Trade Signals</h4>
          <Line data={predSeries} options={commonOptions} />
        </div>
      );
    }
    
    return null;
  };

  // Filter and paginate models
  const filteredModels = models.filter(model => {
    if (filterType && model.type !== filterType) return false;
    if (filterStatus && model.status !== filterStatus) return false;
    return true;
  });

  const totalPages = Math.ceil(filteredModels.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const paginatedModels = filteredModels.slice(startIndex, startIndex + ITEMS_PER_PAGE);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Model Testing</h3>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Model
          </label>
          
          {/* Filters */}
          <div className="flex gap-4 mb-3">
            <select
              className="p-2 border border-gray-300 rounded-md text-sm"
              value={filterType}
              onChange={(e) => {
                setFilterType(e.target.value);
                setCurrentPage(1);
              }}
            >
              <option value="">All Types</option>
              <option value="AgentNews">News</option>
              <option value="AgentPredTime">Pred Time</option>
              <option value="AgentTradeTime">Trade Time</option>
              <option value="AgentRisk">Risk</option>
              <option value="AgentTradeAggregator">Trade Aggregator</option>
            </select>
            
            <select
              className="p-2 border border-gray-300 rounded-md text-sm"
              value={filterStatus}
              onChange={(e) => {
                setFilterStatus(e.target.value);
                setCurrentPage(1);
              }}
            >
              <option value="">All Statuses</option>
              <option value="SUCCESS">Success</option>
              <option value="FAILURE">Failure</option>
              <option value="PROGRESS">Progress</option>
              <option value="PENDING">Pending</option>
            </select>
          </div>

          {loadingModels ? (
            <div className="flex items-center justify-center p-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="ml-2 text-gray-600">Loading models...</span>
            </div>
          ) : (
            <select
              className="w-full p-2 border border-gray-300 rounded-md"
              value={selectedModel?.id || ''}
              onChange={(e) => {
                const model = models.find(m => m.id === parseInt(e.target.value));
                setSelectedModel(model);
              }}
            >
              <option value="">Choose a model...</option>
              {paginatedModels.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.type}) - {model.status}
                </option>
              ))}
            </select>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-3">
              <div className="text-sm text-gray-600">
                Showing {startIndex + 1}-{Math.min(startIndex + ITEMS_PER_PAGE, filteredModels.length)} of {filteredModels.length} models
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-1 text-sm border border-gray-300 rounded disabled:opacity-50"
                >
                  Previous
                </button>
                <span className="px-3 py-1 text-sm text-gray-600">
                  {currentPage} / {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                  className="px-3 py-1 text-sm border border-gray-300 rounded disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>

        {selectedModel && (
          <div className="mb-4 p-4 bg-gray-50 rounded-md">
            <h4 className="font-medium mb-2">Model Information</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div><span className="font-medium">Type:</span> {selectedModel.type}</div>
              <div><span className="font-medium">Timeframe:</span> {selectedModel.timeframe}</div>
              <div><span className="font-medium">Status:</span> 
                <span className={`ml-1 px-2 py-1 rounded-full text-xs ${getStatusBadge(selectedModel.status)}`}>
                  {selectedModel.status}
                </span>
              </div>
              <div><span className="font-medium">Version:</span> {selectedModel.version}</div>
            </div>
          </div>
        )}

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

        <button
          onClick={onTest}
          disabled={loading || !selectedModel || selectedCoins.length === 0}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Testing...
            </>
          ) : (
            'Start Test'
          )}
        </button>

        {/* Model versions and preview for Risk/TradeTime */}
        {selectedModel && (selectedModel.type === 'AgentRisk' || selectedModel.type === 'AgentTradeTime') && (
          <div className="mt-6 p-4 bg-gray-50 rounded-md">
            <h4 className="font-medium mb-2">Preview Time Series ({selectedModel.type})</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Model Version</label>
                {loadingVersions ? (
                  <div className="flex items-center p-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                    <span className="text-sm text-gray-600">Loading...</span>
                  </div>
                ) : (
                  <select
                    className="w-full p-2 border border-gray-300 rounded-md"
                    value={selectedModelPath}
                    onChange={(e) => setSelectedModelPath(e.target.value)}
                  >
                    <option value="">Choose version...</option>
                    {modelVersions.map((m) => (
                      <option key={m.model_path} value={m.model_path}>{m.model_name}</option>
                    ))}
                  </select>
                )}
              </div>
              <div className="flex items-end">
                <button
                  onClick={onPreviewSeries}
                  disabled={!selectedModelPath || selectedCoins.length === 0 || loadingPreview}
                  className="w-full bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 disabled:bg-gray-400 flex items-center justify-center"
                >
                  {loadingPreview ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Loading...
                    </>
                  ) : (
                    'Preview Series'
                  )}
                </button>
              </div>
            </div>
            <div className="mt-4">
              {renderPreview()}
            </div>
          </div>
        )}

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

      {testResults && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Test Results</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {Object.entries(testResults.test_results || {}).map(([key, value]) => {
              if (typeof value === 'number' && !key.includes('samples') && !key.includes('trades')) {
                return (
                  <div key={key} className="bg-gray-50 p-3 rounded-md hover:bg-gray-100 transition-colors">
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

          {Array.isArray(testResults.test_results?.charts) && testResults.test_results.charts.length > 0 && (
            <div className="mb-6">
              <h4 className="font-medium mb-2">Performance Charts</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {testResults.test_results.charts.map((chart, idx) => (
                  <div key={idx} className="bg-gray-50 p-3 rounded-md">
                    <div className="text-sm font-medium text-gray-700 mb-2">{chart.title}</div>
                    {renderChart(chart)}
                  </div>
                ))}
              </div>
            </div>
          )}

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

          {testResults.test_results?.confusion_matrix && (
            <div className="mb-6">
              <h4 className="font-medium mb-2">Confusion Matrix</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full bg-white border border-gray-200">
                  <thead>
                    <tr>
                      <th className="px-3 py-1 border-b"></th>
                      {testResults.test_results.confusion_matrix.labels.map((l, i) => (
                        <th key={i} className="px-3 py-1 border-b text-xs text-gray-600">Pred {l}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {testResults.test_results.confusion_matrix.data.map((row, i) => (
                      <tr key={i}>
                        <td className="px-3 py-1 border-b text-xs text-gray-600">Actual {testResults.test_results.confusion_matrix.labels[i]}</td>
                        {row.map((val, j) => (
                          <td key={j} className="px-3 py-1 border-b text-xs text-gray-800">{val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

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
