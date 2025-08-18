import React, { useState, useEffect } from 'react';
import { Line, Doughnut } from 'react-chartjs-2';
import BalanceChart from '../charts/BalanceChart';
import AssetsChart from '../charts/AssetsChart';

const FinanceTabContent = ({ 
  user
}) => {
    
  const [assetsData, setAssetsData] = useState(null);

  useEffect(() => {
    setAssetsData({
      crypto: 80,
      cash: 20,
    });
  }, [user.balance]);

  // Настройки графика баланса для темной темы
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        titleColor: '#ffffff',
        bodyColor: '#d1d5db',
        borderColor: '#374151',
        borderWidth: 1,
        padding: 12,
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 13
        },
        callbacks: {
          label: function(context) {
            const datasetLabel = context.dataset.label || '';
            const value = context.parsed.y || 0;
            let label = `${datasetLabel}: $${value.toFixed(2)}`;
            
            if (datasetLabel === 'Forecast') {
              const change = ((value - user.balance) / user.balance * 100).toFixed(2);
              label += ` (${change}%)`;
            }
            
            return label;
          },
          afterLabel: function(context) {
            if (context.datasetIndex === 1) {
              return 'Projected growth based on current trends';
            }
            return null;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        },
        ticks: {
          color: '#9ca3af'
        }
      },
      y: {
        grid: {
          color: 'rgba(156, 163, 175, 0.1)'
        },
        ticks: {
          color: '#9ca3af',
          callback: function(value) {
            return '$' + value;
          }
        }
      }
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
    hover: {
      mode: 'index',
      intersect: false
    }
  };

  // Данные для графика
  const getChartData = () => {
    const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    // Фактические данные
    const actualData = Array(6).fill(null)
      .map((_, i) => user.balance * (0.8 + Math.random() * 0.4))
      .concat(Array(6).fill(null));
    
    // Прогнозные данные
    const forecastData = Array(6).fill(null)
      .concat(Array(6).fill(null)
        .map((_, i) => user.balance * (1.1 + i * 0.05 + Math.random() * 0.1)));
    
    return {
      labels,
      datasets: [
        {
          label: 'Actual',
          data: actualData,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 3,
          fill: true,
          tension: 0.4,
          pointBackgroundColor: '#3b82f6',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8
        },
        {
          label: 'Forecast',
          data: forecastData,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          borderWidth: 3,
          borderDash: [5, 5],
          fill: false,
          tension: 0.4,
          pointBackgroundColor: '#10b981',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8
        }
      ]
    };
  };

  // Настройки для круговой диаграммы активов
  const assetsChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#d1d5db',
          padding: 20,
          usePointStyle: true,
          pointStyle: 'circle'
        }
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        titleColor: '#ffffff',
        bodyColor: '#d1d5db',
        borderColor: '#374151',
        borderWidth: 1
      }
    }
  };

  const getAssetsChartData = () => ({
    labels: ['Криптовалюты', 'Наличные'],
    datasets: [{
      data: [assetsData?.crypto || 0, assetsData?.cash || 0],
      backgroundColor: [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)'
      ],
      borderColor: [
        'rgba(59, 130, 246, 1)',
        'rgba(16, 185, 129, 1)'
      ],
      borderWidth: 2,
      hoverBackgroundColor: [
        'rgba(59, 130, 246, 1)',
        'rgba(16, 185, 129, 1)'
      ]
    }]
  });

  return (
    <div className="w-full space-y-8">
      {/* Основные финансовые показатели */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 border border-blue-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 1.343-3 3s1.343 3 3 3 3-1.343 3-3-1.343-3-3-3z"></path>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 1v3m0 16v3m8.485-10.485l-2.121-2.121M5.636 18.364l-2.121-2.121M21 12h-3M6 12H3m15.364 5.364l-2.121 2.121M7.757 5.636L5.636 3.515"></path>
              </svg>
            </div>
            <div className="text-right">
              <div className="text-sm text-blue-300">Текущий баланс</div>
              <div className="text-2xl font-bold text-white">${user.balance?.toFixed(2) || '0.00'}</div>
            </div>
          </div>
          <div className="flex items-center text-green-400 text-sm">
            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
            </svg>
            +12.5%
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 border border-green-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-green-600 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
              </svg>
            </div>
            <div className="text-right">
              <div className="text-sm text-green-300">Общая прибыль</div>
              <div className="text-2xl font-bold text-white">$2,847.50</div>
            </div>
          </div>
          <div className="flex items-center text-green-400 text-sm">
            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
            </svg>
            +8.3%
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/20 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
              </svg>
            </div>
            <div className="text-right">
              <div className="text-sm text-purple-300">Активные сделки</div>
              <div className="text-2xl font-bold text-white">24</div>
            </div>
          </div>
          <div className="flex items-center text-purple-400 text-sm">
            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
            </svg>
            +3 новых
          </div>
        </div>

        <div className="bg-gradient-to-br from-yellow-500/20 to-yellow-600/20 border border-yellow-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-yellow-500 to-yellow-600 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
            </div>
            <div className="text-right">
              <div className="text-sm text-yellow-300">Точность</div>
              <div className="text-2xl font-bold text-white">89%</div>
            </div>
          </div>
          <div className="flex items-center text-yellow-400 text-sm">
            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
            </svg>
            +2.1%
          </div>
        </div>
      </div>

      {/* Графики */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* График баланса */}
        <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 backdrop-blur-sm rounded-2xl border border-gray-600/30 p-6 shadow-xl">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"></path>
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white">Динамика баланса</h3>
          </div>
          <div className="h-80">
            <Line data={getChartData()} options={chartOptions} />
          </div>
        </div>

        {/* Круговая диаграмма активов */}
        <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 backdrop-blur-sm rounded-2xl border border-gray-600/30 p-6 shadow-xl">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-green-600 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white">Распределение активов</h3>
          </div>
          <div className="h-80">
            <Doughnut data={getAssetsChartData()} options={assetsChartOptions} />
          </div>
        </div>
      </div>

      {/* Последние транзакции */}
      <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 backdrop-blur-sm rounded-2xl border border-gray-600/30 p-6 shadow-xl">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
            </svg>
          </div>
          <h3 className="text-xl font-bold text-white">Последние транзакции</h3>
        </div>
        
        <div className="space-y-4">
          {[
            { type: 'buy', asset: 'BTC', amount: 0.5, price: 45000, time: '2 часа назад', status: 'completed' },
            { type: 'sell', asset: 'ETH', amount: 2.1, price: 3200, time: '4 часа назад', status: 'completed' },
            { type: 'buy', asset: 'ADA', amount: 1000, price: 0.45, time: '6 часов назад', status: 'pending' },
            { type: 'sell', asset: 'DOT', amount: 50, price: 18.5, time: '8 часов назад', status: 'completed' }
          ].map((transaction, index) => (
            <div key={index} className="flex items-center justify-between p-4 bg-gray-700/30 rounded-xl border border-gray-600/30">
              <div className="flex items-center space-x-4">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                  transaction.type === 'buy' 
                    ? 'bg-gradient-to-r from-green-500 to-green-600' 
                    : 'bg-gradient-to-r from-red-500 to-red-600'
                }`}>
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    {transaction.type === 'buy' ? (
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                    ) : (
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"></path>
                    )}
                  </svg>
                </div>
                <div>
                  <div className="text-white font-medium">
                    {transaction.type === 'buy' ? 'Покупка' : 'Продажа'} {transaction.asset}
                  </div>
                  <div className="text-gray-400 text-sm">{transaction.time}</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-white font-medium">
                  {transaction.amount} {transaction.asset}
                </div>
                <div className="text-gray-400 text-sm">
                  ${(transaction.amount * transaction.price).toFixed(2)}
                </div>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                transaction.status === 'completed' 
                  ? 'bg-green-500/20 text-green-300 border border-green-500/30' 
                  : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
              }`}>
                {transaction.status === 'completed' ? 'Завершено' : 'В обработке'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FinanceTabContent;

