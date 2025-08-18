import React, { useState } from 'react';
import { mlService } from '../../services/mlService';

const DataManager = () => {
  const [selectedCoins, setSelectedCoins] = useState([]);
  const [timeframe, setTimeframe] = useState('5m');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [dataStats, setDataStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [importLoading, setImportLoading] = useState(false);

  const onGetStats = async () => {
    setLoading(true);
    try {
      const params = {
        timeframe: timeframe,
        start_date: startDate,
        end_date: endDate
      };
      
      if (selectedCoins.length > 0) {
        params.coins = selectedCoins.join(',');
      }
      
      const stats = await mlService.getDataStats(params);
      setDataStats(stats);
    } catch (error) {
      console.error('Error getting data stats:', error);
      alert('Error getting data statistics');
    } finally {
      setLoading(false);
    }
  };

  const onExport = async () => {
    if (selectedCoins.length === 0) {
      alert('Please select at least one coin');
      return;
    }
    
    if (!startDate || !endDate) {
      alert('Please select start and end dates');
      return;
    }

    try {
      const params = {
        coins: selectedCoins.join(','),
        timeframe: timeframe,
        start_date: startDate,
        end_date: endDate,
        format: 'csv'
      };
      
      const response = await mlService.exportData(params);
      
      // Create download link
      const blob = new Blob([response], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trading_data_${startDate}_${endDate}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting data:', error);
      alert('Error exporting data');
    }
  };

  const onImport = async () => {
    if (!selectedFile) {
      alert('Please select a file');
      return;
    }

    setImportLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('timeframe', timeframe);
      
      const result = await mlService.importData(formData);
      alert(`Import completed: ${result.imported_records} records imported, ${result.skipped_records} skipped`);
      setSelectedFile(null);
    } catch (error) {
      console.error('Error importing data:', error);
      alert('Error importing data');
    } finally {
      setImportLoading(false);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      const allowedTypes = ['text/csv', 'application/json'];
      if (!allowedTypes.includes(file.type)) {
        alert('Please select a CSV or JSON file');
        return;
      }
      setSelectedFile(file);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Data Management</h3>
        
        {/* Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Coins (comma-separated IDs)
            </label>
            <input
              type="text"
              className="w-full p-2 border border-gray-300 rounded-md"
              placeholder="1,2,3 (leave empty for all coins)"
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

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-4 mb-6">
          <button
            onClick={onGetStats}
            disabled={loading}
            className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? 'Loading...' : 'Get Statistics'}
          </button>
          
          <button
            onClick={onExport}
            disabled={selectedCoins.length === 0 || !startDate || !endDate}
            className="bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-400"
          >
            Export Data
          </button>
        </div>

        {/* File Upload */}
        <div className="border-t pt-6">
          <h4 className="font-medium mb-4">Import Data</h4>
          <div className="flex items-center gap-4">
            <input
              type="file"
              accept=".csv,.json"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            <button
              onClick={onImport}
              disabled={!selectedFile || importLoading}
              className="bg-purple-600 text-white py-2 px-4 rounded-md hover:bg-purple-700 disabled:bg-gray-400"
            >
              {importLoading ? 'Importing...' : 'Import'}
            </button>
          </div>
          {selectedFile && (
            <p className="text-sm text-gray-600 mt-2">
              Selected: {selectedFile.name}
            </p>
          )}
        </div>
      </div>

      {/* Data Statistics */}
      {dataStats && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Data Statistics</h3>
          
          {/* Overall Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-blue-50 p-4 rounded-md">
              <div className="text-sm font-medium text-blue-600">Total Records</div>
              <div className="text-2xl font-bold text-blue-900">{dataStats.total_records?.toLocaleString()}</div>
            </div>
            
            <div className="bg-green-50 p-4 rounded-md">
              <div className="text-sm font-medium text-green-600">Coins Count</div>
              <div className="text-2xl font-bold text-green-900">{dataStats.coins_count}</div>
            </div>
            
            <div className="bg-yellow-50 p-4 rounded-md">
              <div className="text-sm font-medium text-yellow-600">Completeness</div>
              <div className="text-2xl font-bold text-yellow-900">
                {dataStats.completeness ? `${(dataStats.completeness * 100).toFixed(1)}%` : 'N/A'}
              </div>
            </div>
            
            <div className="bg-red-50 p-4 rounded-md">
              <div className="text-sm font-medium text-red-600">Missing Values</div>
              <div className="text-2xl font-bold text-red-900">{dataStats.missing_values?.toLocaleString()}</div>
            </div>
          </div>

          {/* Date Range */}
          <div className="mb-6">
            <h4 className="font-medium mb-2">Date Range</h4>
            <div className="bg-gray-50 p-3 rounded-md">
              <div className="text-sm text-gray-600">{dataStats.date_range}</div>
            </div>
          </div>

          {/* Sample Data */}
          {dataStats.sample_data && dataStats.sample_data.length > 0 && (
            <div className="mb-6">
              <h4 className="font-medium mb-2">Sample Data</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full bg-white border border-gray-200">
                  <thead>
                    <tr>
                      <th className="px-4 py-2 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        DateTime
                      </th>
                      <th className="px-4 py-2 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Coin
                      </th>
                      <th className="px-4 py-2 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Open
                      </th>
                      <th className="px-4 py-2 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        High
                      </th>
                      <th className="px-4 py-2 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Low
                      </th>
                      <th className="px-4 py-2 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Close
                      </th>
                      <th className="px-4 py-2 border-b text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Volume
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataStats.sample_data.map((record, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-4 py-2 border-b text-sm text-gray-900">
                          {new Date(record.datetime).toLocaleString()}
                        </td>
                        <td className="px-4 py-2 border-b text-sm text-gray-900">{record.coin}</td>
                        <td className="px-4 py-2 border-b text-sm text-gray-900">{record.open}</td>
                        <td className="px-4 py-2 border-b text-sm text-gray-900">{record.high}</td>
                        <td className="px-4 py-2 border-b text-sm text-gray-900">{record.low}</td>
                        <td className="px-4 py-2 border-b text-sm text-gray-900">{record.close}</td>
                        <td className="px-4 py-2 border-b text-sm text-gray-900">{record.volume?.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Coins Details */}
          {dataStats.coins_details && dataStats.coins_details.length > 0 && (
            <div>
              <h4 className="font-medium mb-2">Coins Details</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {dataStats.coins_details.map((coin, index) => (
                  <div key={index} className="bg-gray-50 p-4 rounded-md">
                    <h5 className="font-medium text-gray-900 mb-2">{coin.name}</h5>
                    <div className="space-y-1 text-sm text-gray-600">
                      <div>Records: {coin.records?.toLocaleString()}</div>
                      <div>Completeness: {coin.completeness ? `${(coin.completeness * 100).toFixed(1)}%` : 'N/A'}</div>
                      <div>Missing: {coin.missing?.toLocaleString()}</div>
                      {coin.first_record && (
                        <div>First: {new Date(coin.first_record).toLocaleDateString()}</div>
                      )}
                      {coin.last_record && (
                        <div>Last: {new Date(coin.last_record).toLocaleDateString()}</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DataManager;
