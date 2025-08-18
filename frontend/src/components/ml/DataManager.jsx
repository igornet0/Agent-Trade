import React, { useEffect, useState } from 'react';
import { get_coins } from '../../services/strategyService';
import { getDataStats, exportData, importData } from '../../services/mlService';

export default function DataManager() {
  const [coins, setCoins] = useState([]);
  const [selectedCoinIds, setSelectedCoinIds] = useState([]);
  const [timeframe, setTimeframe] = useState('5m');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [dataStats, setDataStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [importing, setImporting] = useState(false);

  const TIMEFRAMES = ['5m','15m','30m','1h','4h','1d'];

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

  const toggleCoin = (id) => {
    setSelectedCoinIds((prev) => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const onGetStats = async () => {
    if (selectedCoinIds.length === 0) return;
    setLoading(true);
    try {
      const stats = await getDataStats({
        coins: selectedCoinIds,
        timeframe,
        start_date: startDate,
        end_date: endDate
      });
      setDataStats(stats);
    } catch (e) {
      console.error(e);
      alert('Ошибка получения статистики');
    } finally {
      setLoading(false);
    }
  };

  const onExport = async () => {
    if (selectedCoinIds.length === 0) return;
    setExporting(true);
    try {
      const result = await exportData({
        coins: selectedCoinIds,
        timeframe,
        start_date: startDate,
        end_date: endDate,
        format: 'csv'
      });
      
      // Создаем ссылку для скачивания
      const blob = new Blob([result.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trading_data_${startDate}_${endDate}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      alert('Данные экспортированы');
    } catch (e) {
      console.error(e);
      alert('Ошибка экспорта данных');
    } finally {
      setExporting(false);
    }
  };

  const onImport = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setImporting(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('timeframe', timeframe);
      
      await importData(formData);
      alert('Данные импортированы');
    } catch (e) {
      console.error(e);
      alert('Ошибка импорта данных');
    } finally {
      setImporting(false);
    }
  };

  const formatNumber = (num) => {
    return new Intl.NumberFormat().format(num);
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleString();
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Управление данными</h2>
          <p className="text-sm text-gray-600 mt-1">Просмотр, экспорт и импорт данных для обучения и тестирования</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
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
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Начальная дата</label>
          <input 
            type="date" 
            value={startDate} 
            onChange={(e)=>setStartDate(e.target.value)} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Конечная дата</label>
          <input 
            type="date" 
            value={endDate} 
            onChange={(e)=>setEndDate(e.target.value)} 
            className="w-full p-3 border border-gray-300 rounded-lg" 
          />
        </div>
        <div className="flex items-end">
          <button 
            onClick={onGetStats} 
            disabled={loading || selectedCoinIds.length === 0}
            className={`w-full px-4 py-3 rounded-md font-medium ${
              (loading || selectedCoinIds.length === 0) 
                ? 'bg-gray-300 cursor-not-allowed' 
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {loading ? 'Загрузка...' : 'Получить статистику'}
          </button>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Выберите монеты</label>
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

      <div className="flex space-x-4">
        <button 
          onClick={onExport} 
          disabled={exporting || selectedCoinIds.length === 0}
          className={`px-4 py-2 rounded-md font-medium ${
            (exporting || selectedCoinIds.length === 0) 
              ? 'bg-gray-300 cursor-not-allowed' 
              : 'bg-green-600 text-white hover:bg-green-700'
          }`}
        >
          {exporting ? 'Экспорт...' : 'Экспорт данных'}
        </button>
        
        <label className={`px-4 py-2 rounded-md font-medium cursor-pointer ${
          importing 
            ? 'bg-gray-300 cursor-not-allowed' 
            : 'bg-purple-600 text-white hover:bg-purple-700'
        }`}>
          {importing ? 'Импорт...' : 'Импорт данных'}
          <input 
            type="file" 
            accept=".csv,.json" 
            onChange={onImport} 
            disabled={importing}
            className="hidden" 
          />
        </label>
      </div>

      {dataStats && (
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Статистика данных</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white p-4 rounded border">
              <h4 className="text-sm font-medium text-gray-600 mb-2">Общая информация</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Всего записей:</span>
                  <span className="font-medium">{formatNumber(dataStats.total_records)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Монет:</span>
                  <span className="font-medium">{dataStats.coins_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Период:</span>
                  <span className="font-medium">{dataStats.date_range}</span>
                </div>
              </div>
            </div>

            <div className="bg-white p-4 rounded border">
              <h4 className="text-sm font-medium text-gray-600 mb-2">Качество данных</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Полнота:</span>
                  <span className="font-medium">{(dataStats.completeness * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Пропуски:</span>
                  <span className="font-medium">{formatNumber(dataStats.missing_values)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Дубликаты:</span>
                  <span className="font-medium">{formatNumber(dataStats.duplicates)}</span>
                </div>
              </div>
            </div>

            <div className="bg-white p-4 rounded border">
              <h4 className="text-sm font-medium text-gray-600 mb-2">Временные характеристики</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Первая запись:</span>
                  <span className="font-medium text-xs">{formatDate(dataStats.first_record)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Последняя запись:</span>
                  <span className="font-medium text-xs">{formatDate(dataStats.last_record)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Интервал:</span>
                  <span className="font-medium">{dataStats.timeframe}</span>
                </div>
              </div>
            </div>
          </div>

          {dataStats.coins_details && (
            <div className="mt-6">
              <h4 className="text-md font-semibold text-gray-900 mb-3">Детали по монетам</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full bg-white border rounded-lg">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Монета</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Записей</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Полнота</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Пропуски</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Первая запись</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Последняя запись</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {dataStats.coins_details.map((coin, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-4 py-2 text-sm font-medium text-gray-900">{coin.name}</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{formatNumber(coin.records)}</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{(coin.completeness * 100).toFixed(1)}%</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{formatNumber(coin.missing)}</td>
                        <td className="px-4 py-2 text-sm text-gray-600 text-xs">{formatDate(coin.first_record)}</td>
                        <td className="px-4 py-2 text-sm text-gray-600 text-xs">{formatDate(coin.last_record)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {dataStats.sample_data && (
            <div className="mt-6">
              <h4 className="text-md font-semibold text-gray-900 mb-3">Пример данных</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full bg-white border rounded-lg">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Дата</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Монета</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Open</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">High</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Low</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Close</th>
                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Volume</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {dataStats.sample_data.map((row, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-4 py-2 text-sm text-gray-600 text-xs">{formatDate(row.datetime)}</td>
                        <td className="px-4 py-2 text-sm font-medium text-gray-900">{row.coin}</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{row.open}</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{row.high}</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{row.low}</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{row.close}</td>
                        <td className="px-4 py-2 text-sm text-gray-600">{formatNumber(row.volume)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
