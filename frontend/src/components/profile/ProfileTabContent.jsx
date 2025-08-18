import React from 'react';

const ProfileTabContent = ({ user }) => {
  return (
    <div className="w-full">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Основные данные */}
        <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 backdrop-blur-sm rounded-2xl border border-gray-600/30 p-6 shadow-xl">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white">Основные данные</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Имя пользователя</label>
              <div className="p-4 bg-gray-700/50 border border-gray-600/50 rounded-xl text-white backdrop-blur-sm">
                {user.login}
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
              <div className="p-4 bg-gray-700/50 border border-gray-600/50 rounded-xl text-white backdrop-blur-sm">
                {user.email}
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Дата регистрации</label>
              <div className="p-4 bg-gray-700/50 border border-gray-600/50 rounded-xl text-white backdrop-blur-sm">
                {user.created.split('T')[0]}
              </div>
            </div>
          </div>
        </div>
        
        {/* Безопасность */}
        <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 backdrop-blur-sm rounded-2xl border border-gray-600/30 p-6 shadow-xl">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-green-600 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white">Безопасность</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Статус аккаунта</label>
              <div className="p-4 bg-gradient-to-r from-green-500/20 to-green-600/20 border border-green-500/30 text-green-300 rounded-xl backdrop-blur-sm flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="font-medium">Подтвержден</span>
              </div>
            </div>
            <button className="w-full bg-gradient-to-r from-blue-500/20 to-blue-600/20 hover:from-blue-500/30 hover:to-blue-600/30 text-blue-300 hover:text-blue-200 p-4 rounded-xl transition-all duration-300 border border-blue-500/30 hover:border-blue-400/50 backdrop-blur-sm font-medium">
              Сменить пароль
            </button>
            <button className="w-full bg-gradient-to-r from-purple-500/20 to-purple-600/20 hover:from-purple-500/30 hover:to-purple-600/30 text-purple-300 hover:text-purple-200 p-4 rounded-xl transition-all duration-300 border border-purple-500/30 hover:border-purple-400/50 backdrop-blur-sm font-medium">
              Двухфакторная аутентификация
            </button>
          </div>
        </div>
      </div>

      {/* Статистика */}
      <div className="mt-8 bg-gradient-to-br from-gray-800/50 to-gray-700/50 backdrop-blur-sm rounded-2xl border border-gray-600/30 p-6 shadow-xl">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-r from-yellow-500 to-yellow-600 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
          </div>
          <h3 className="text-xl font-bold text-white">Статистика активности</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gradient-to-r from-blue-500/10 to-blue-600/10 border border-blue-500/20 rounded-xl p-4 text-center">
            <div className="text-2xl font-bold text-blue-300 mb-1">24</div>
            <div className="text-sm text-gray-400">Активных агентов</div>
          </div>
          <div className="bg-gradient-to-r from-green-500/10 to-green-600/10 border border-green-500/20 rounded-xl p-4 text-center">
            <div className="text-2xl font-bold text-green-300 mb-1">156</div>
            <div className="text-sm text-gray-400">Успешных сделок</div>
          </div>
          <div className="bg-gradient-to-r from-purple-500/10 to-purple-600/10 border border-purple-500/20 rounded-xl p-4 text-center">
            <div className="text-2xl font-bold text-purple-300 mb-1">89%</div>
            <div className="text-sm text-gray-400">Точность прогнозов</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfileTabContent;

