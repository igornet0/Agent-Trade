import React, { useState } from 'react';
import ProfileTabContent from '../components/profile/ProfileTabContent';
import FinanceTabContent from '../components/profile/FinanceTabContent';
import AgentsTabContent from '../components/profile/AgentsTabContent';
import ModelsTabContent from '../components/profile/ModelsTabContent';
import StrategyTabContent from '../components/profile/StrategyTabContent';
import StrategyTable from '../components/profile/StrategyTable';
import CoinsTabContent from '../components/profile/CoinsTabContent';
import ModuleTester from '../components/profile/ModuleTester';

const ProfileSidebar = ({ activeTab, setActiveTab, onLogout }) => {
  const navItems = [
    { key: 'profile', label: 'Профиль', icon: (
      <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" d="M5.121 17.804A13.937 13.937 0 0112 15c2.485 0 4.847.636 6.879 1.804M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 14c-4.418 0-8 1.79-8 4v1h16v-1c0-2.21-3.582-4-8-4z"></path>
      </svg>
    ) },
    { key: 'finance', label: 'Финансы', icon: (
      <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 1.343-3 3s1.343 3 3 3 3-1.343 3-3-1.343-3-3-3z"></path>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 1v3m0 16v3m8.485-10.485l-2.121-2.121M5.636 18.364l-2.121-2.121M21 12h-3M6 12H3m15.364 5.364l-2.121 2.121M7.757 5.636L5.636 3.515"></path>
      </svg>
    ) },
    { key: 'agents', label: 'Агенты', icon: (
      <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a4 4 0 00-3-3.87"></path>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 20H4v-2a4 4 0 013-3.87"></path>
        <circle cx="12" cy="7" r="4" strokeLinecap="round" strokeLinejoin="round"></circle>
      </svg>
    ) },
    { key: 'models', label: 'Модели (ML Studio)', icon: (
      <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <rect width="14" height="14" x="5" y="5" rx="2" ry="2"></rect>
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 10h14"></path>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 5v14"></path>
      </svg>
    ) },
    { key: 'pipeline', label: 'Пайплайн', icon: (
      <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v4a1 1 0 001 1h3v3a1 1 0 001 1h4"></path>
        <path strokeLinecap="round" strokeLinejoin="round" d="M21 3l-6 6"></path>
        <path strokeLinecap="round" strokeLinejoin="round" d="M14 3h7v7"></path>
      </svg>
    ) },
    { key: 'strategy', label: 'Стратегия', icon: (
      <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v4a1 1 0 001 1h3v3a1 1 0 001 1h4"></path>
        <path strokeLinecap="round" strokeLinejoin="round" d="M21 3l-6 6"></path>
        <path strokeLinecap="round" strokeLinejoin="round" d="M14 3h7v7"></path>
      </svg>
    ) },
    { key: 'strategys', label: 'Таблица', icon: (
      <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <rect x="3" y="4" width="18" height="4" rx="1"></rect>
        <rect x="3" y="10" width="18" height="4" rx="1"></rect>
        <rect x="3" y="16" width="18" height="4" rx="1"></rect>
      </svg>
    ) },
    { key: 'coins', label: 'Монеты', icon: (
      <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10"></circle>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6l4 2"></path>
      </svg>
    ) },
    { key: 'module_tester', label: 'Тестирование', icon: (
      <svg className="w-5 h-5 mb-1" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"></path>
      </svg>
    ) },
  ];

  return (
    <div className="w-20 bg-[#141e1b] flex flex-col items-center pt-4">
      <div className="mb-8">
        <div className="text-4xl text-[#00ff75]">🔷</div>
      </div>
      {navItems.map(({ key, label, icon }) => (
        <button
          key={key}
          onClick={() => setActiveTab(key)}
          className={`w-16 h-16 bg-[#00ff75] rounded-[30%] flex flex-col items-center justify-center my-2 text-black ${
            activeTab === key ? 'ring-2 ring-white' : ''
          }`}
          title={label}
        >
          {icon}
          <span className="text-xs">{label}</span>
        </button>
      ))}
      <div className="mt-auto mb-4">
        <button
          onClick={onLogout}
          className="w-16 h-16 bg-[#00ff75] rounded-[30%] flex flex-col items-center justify-center my-2 text-black"
          title="Выход"
        >
          <span className="mb-1 text-2xl">🚪</span>
          <span className="text-xs">Выход</span>
        </button>
      </div>
    </div>
  );
};

const ProfilePage = ({ user, onLogout }) => {
  const [activeTab, setActiveTab] = useState('profile');
  return (
    <div className="min-h-screen bg-gray-100">
      <div className="flex">
        <ProfileSidebar activeTab={activeTab} setActiveTab={setActiveTab} onLogout={onLogout} />
        <div className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
          {activeTab === 'profile' && <ProfileTabContent user={user} />}
          {activeTab === 'finance' && <FinanceTabContent />}
          {activeTab === 'agents' && <AgentsTabContent />}
          {activeTab === 'models' && <ModelsTabContent />}
          {activeTab === 'pipeline' && <StrategyTabContent />}
          {activeTab === 'strategy' && <StrategyTabContent />}
          {activeTab === 'strategys' && <StrategyTable />}
          {activeTab === 'coins' && <CoinsTabContent />}
          {activeTab === 'module_tester' && <ModuleTester />}
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;