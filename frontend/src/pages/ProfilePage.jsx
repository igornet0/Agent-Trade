import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ProfileTabContent from '../components/profile/ProfileTabContent';
import FinanceTabContent from '../components/profile/FinanceTabContent';
import AgentsTabContent from '../components/profile/AgentsTabContent';
import ModelsTabContent from '../components/profile/ModelsTabContent';
import StrategyTabContent from '../components/profile/StrategyTabContent';
import StrategyTable from '../components/profile/StrategyTable';
import CoinsTabContent from '../components/profile/CoinsTabContent';
import ModuleTester from '../components/profile/ModuleTester';
import PipelineBuilder from '../components/pipeline/PipelineBuilder';

const ProfileSidebar = ({ activeTab, setActiveTab, onLogout }) => {
  const navigate = useNavigate();
  
  const navItems = [
    { 
      key: 'profile', 
      label: 'Профиль', 
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
        </svg>
      ),
      color: 'from-blue-500 to-blue-600'
    },
    { 
      key: 'finance', 
      label: 'Финансы', 
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 1.343-3 3s1.343 3 3 3 3-1.343 3-3-1.343-3-3-3z"></path>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 1v3m0 16v3m8.485-10.485l-2.121-2.121M5.636 18.364l-2.121-2.121M21 12h-3M6 12H3m15.364 5.364l-2.121 2.121M7.757 5.636L5.636 3.515"></path>
        </svg>
      ),
      color: 'from-green-500 to-green-600'
    },
    { 
      key: 'agents', 
      label: 'Агенты', 
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a4 4 0 00-3-3.87"></path>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 20H4v-2a4 4 0 013-3.87"></path>
          <circle cx="12" cy="7" r="4" strokeLinecap="round" strokeLinejoin="round"></circle>
        </svg>
      ),
      color: 'from-purple-500 to-purple-600'
    },
    { 
      key: 'models', 
      label: 'ML Studio', 
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
        </svg>
      ),
      color: 'from-indigo-500 to-indigo-600'
    },
    { 
      key: 'pipeline', 
      label: 'Пайплайн', 
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
        </svg>
      ),
      color: 'from-yellow-500 to-yellow-600'
    },
    { 
      key: 'strategy', 
      label: 'Стратегия', 
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
        </svg>
      ),
      color: 'from-red-500 to-red-600'
    },
    { 
      key: 'strategys', 
      label: 'Таблица', 
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <rect x="3" y="4" width="18" height="4" rx="1"></rect>
          <rect x="3" y="10" width="18" height="4" rx="1"></rect>
          <rect x="3" y="16" width="18" height="4" rx="1"></rect>
        </svg>
      ),
      color: 'from-pink-500 to-pink-600'
    },
    { 
      key: 'coins', 
      label: 'Монеты', 
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 1.343-3 3s1.343 3 3 3 3-1.343 3-3-1.343-3-3-3z"></path>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 1v3m0 16v3m8.485-10.485l-2.121-2.121M5.636 18.364l-2.121-2.121M21 12h-3M6 12H3m15.364 5.364l-2.121 2.121M7.757 5.636L5.636 3.515"></path>
        </svg>
      ),
      color: 'from-orange-500 to-orange-600'
    },
    { 
      key: 'module_tester', 
      label: 'Тестирование', 
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
        </svg>
      ),
      color: 'from-teal-500 to-teal-600'
    },
  ];

  return (
    <div className="w-80 bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 border-r border-gray-700 flex flex-col h-screen">
      {/* Logo Header */}
      <div className="p-6 border-b border-gray-700">
        <button
          onClick={() => navigate('/')}
          className="flex items-center space-x-3 group transition-all duration-300 hover:scale-105"
        >
          <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
            </svg>
          </div>
          <div className="text-left">
            <h1 className="text-xl font-bold text-white group-hover:text-blue-300 transition-colors duration-300">
              AI Trading
            </h1>
            <p className="text-sm text-gray-400 group-hover:text-gray-300 transition-colors duration-300">
              Панель управления
            </p>
          </div>
        </button>
      </div>

      {/* Navigation Items */}
      <div className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
        {navItems.map(({ key, label, icon, color }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`w-full group relative transition-all duration-300 ${
              activeTab === key 
                ? 'transform scale-105' 
                : 'hover:scale-102'
            }`}
          >
            <div className={`relative overflow-hidden rounded-xl p-4 transition-all duration-300 ${
              activeTab === key
                ? `bg-gradient-to-r ${color} shadow-lg shadow-${color.split('-')[1]}-500/25`
                : 'bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700/50 hover:border-gray-600/50'
            }`}>
              {/* Active indicator */}
              {activeTab === key && (
                <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent rounded-xl"></div>
              )}
              
              <div className="relative flex items-center space-x-4">
                <div className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-300 ${
                  activeTab === key
                    ? 'bg-white/20 text-white'
                    : 'bg-gray-700/50 text-gray-400 group-hover:bg-gray-600/50 group-hover:text-gray-300'
                }`}>
                  {icon}
                </div>
                <div className="flex-1 text-left">
                  <span className={`font-medium transition-all duration-300 ${
                    activeTab === key
                      ? 'text-white'
                      : 'text-gray-300 group-hover:text-white'
                  }`}>
                    {label}
                  </span>
                </div>
                {activeTab === key && (
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                )}
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Logout Button */}
      <div className="p-4 border-t border-gray-700">
        <button
          onClick={onLogout}
          className="w-full group relative overflow-hidden rounded-xl p-4 bg-gradient-to-r from-red-500/20 to-red-600/20 hover:from-red-500/30 hover:to-red-600/30 border border-red-500/30 hover:border-red-400/50 transition-all duration-300 transform hover:scale-105"
        >
          <div className="flex items-center space-x-4">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-red-500/20 flex items-center justify-center text-red-400 group-hover:text-red-300 transition-colors duration-300">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path>
              </svg>
            </div>
            <div className="flex-1 text-left">
              <span className="font-medium text-red-300 group-hover:text-red-200 transition-colors duration-300">
                Выход
              </span>
            </div>
          </div>
        </button>
      </div>
    </div>
  );
};

const ProfilePage = ({ user, onLogout }) => {
  const [activeTab, setActiveTab] = useState('profile');

  useEffect(() => {
    const mapHashToTab = (hash) => {
      const h = (hash || '').replace('#', '');
      if (!h) return null;
      // allow known keys direct mapping
      const allowed = new Set(['profile','finance','agents','models','pipeline','strategy','strategys','coins','module_tester']);
      return allowed.has(h) ? h : null;
    };
    const applyHash = () => {
      const tab = mapHashToTab(window.location.hash);
      if (tab) setActiveTab(tab);
    };
    applyHash();
    window.addEventListener('hashchange', applyHash);
    return () => window.removeEventListener('hashchange', applyHash);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <div className="flex h-screen">
        <ProfileSidebar activeTab={activeTab} setActiveTab={setActiveTab} onLogout={onLogout} />
        <div className="flex-1 overflow-auto">
          <div className="p-8">
            {/* Page Header */}
            <div className="mb-8">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold text-white mb-2">
                    {activeTab === 'profile' && 'Профиль пользователя'}
                    {activeTab === 'finance' && 'Финансовый анализ'}
                    {activeTab === 'agents' && 'Торговые агенты'}
                    {activeTab === 'models' && 'ML Studio'}
                    {activeTab === 'pipeline' && 'Пайплайн'}
                    {activeTab === 'strategy' && 'Стратегии'}
                    {activeTab === 'strategys' && 'Таблица стратегий'}
                    {activeTab === 'coins' && 'Криптовалюты'}
                    {activeTab === 'module_tester' && 'Тестирование модулей'}
                  </h1>
                  <p className="text-gray-400">
                    {activeTab === 'profile' && 'Управление профилем и настройками'}
                    {activeTab === 'finance' && 'Анализ финансовых показателей и транзакций'}
                    {activeTab === 'agents' && 'Мониторинг и управление торговыми агентами'}
                    {activeTab === 'models' && 'Создание и обучение моделей машинного обучения'}
                    {activeTab === 'pipeline' && 'Настройка торговых пайплайнов'}
                    {activeTab === 'strategy' && 'Создание и управление торговыми стратегиями'}
                    {activeTab === 'strategys' && 'Просмотр всех стратегий в табличном виде'}
                    {activeTab === 'coins' && 'Анализ криптовалют и их динамики'}
                    {activeTab === 'module_tester' && 'Тестирование различных модулей системы'}
                  </p>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm font-medium">Система активна</span>
                </div>
              </div>
            </div>

            {/* Content Area */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700/50 p-6 shadow-xl">
              {activeTab === 'profile' && <ProfileTabContent user={user} />}
              {activeTab === 'finance' && <FinanceTabContent />}
              {activeTab === 'agents' && <AgentsTabContent />}
              {activeTab === 'models' && <ModelsTabContent />}
              {activeTab === 'pipeline' && <PipelineBuilder />}
              {activeTab === 'strategy' && <StrategyTabContent />}
              {activeTab === 'strategys' && <StrategyTable />}
              {activeTab === 'coins' && <CoinsTabContent />}
              {activeTab === 'module_tester' && <ModuleTester />}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;