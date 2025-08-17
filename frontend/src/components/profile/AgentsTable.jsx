import React, { useState } from 'react';
import { FaCheckCircle, FaExclamationTriangle } from 'react-icons/fa';
import { get_train_status } from '../../services/strategyService';
import { promoteModel } from '../../services/mlService';

const AgentsTable = ({ agents, onAgentClick, onTrainNewAgent, onEvaluateAgent }) => {
  const [isTraining, setIsTraining] = useState(false);

  const handleCheckProgress = async (agentId) => {
    try {
      const s = await get_train_status(agentId);
      alert(`Статус: ${s.status}, epoch: ${s.epoch_now}/${s.epochs}, loss: ${s.loss_now ?? 'n/a'}`);
    } catch (e) { console.error(e); alert('Не удалось получить статус'); }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "train":
        return <div className="animate-spin h-4 w-4 border-t-2 border-blue-500 rounded-full"></div>;
      case "open":
        return <FaCheckCircle className="text-green-500" />;
      case "close":
        return <FaExclamationTriangle className="text-red-500" />;
      default:
        return null;
    }
  };

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Управление агентами</h2>
        <button 
          onClick={onTrainNewAgent}
          className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded flex items-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
          </svg>
          Обучить нового агента
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Версия</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Название</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Тип модели</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Таймфреим</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Статус</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Точность</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Дата создания</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Последнее обучение</th>
              <th className="px-6 py-3"/>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {agents.map((agent) => (
              <tr 
                key={agent.id} 
                className="hover:bg-gray-50"
              >
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{agent.version}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 cursor-pointer" onClick={() => onAgentClick(agent)}>{agent.name}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{agent.type}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{agent.timeframe}</td>
                <td className="px-6 py-4 whitespace-nowrap flex items-center">
                  {getStatusIcon(agent.status)}
                  <span className="ml-2 capitalize">{agent.status}</span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {agent.accuracy ? (agent.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(agent.created).toLocaleDateString()}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {agent.last_trained ? new Date(agent.last_trained).toLocaleDateString() : 'N/A'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-right space-x-2">
                  <button className="px-3 py-1 rounded bg-blue-50 text-blue-700" onClick={()=>handleCheckProgress(agent.id)}>Прогресс</button>
                  <button className="px-3 py-1 rounded bg-indigo-50 text-indigo-700" onClick={()=>onEvaluateAgent && onEvaluateAgent(agent)}>Оценить</button>
                  <button className="px-3 py-1 rounded bg-green-50 text-green-700" onClick={async()=>{ try{ await promoteModel(agent.id); alert('Сделано прод'); }catch(e){ console.error(e); alert('Ошибка промо'); } }}>Сделать прод</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
};

export default AgentsTable;
