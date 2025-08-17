import api from './api';

export const listModels = async ({ type = null } = {}) => {
  const response = await api.get('/api_db_agent/agents/', { params: { type } });
  if (response.status !== 200) throw new Error('Ошибка загрузки моделей');
  return response.data;
};

export const trainModel = async (agentType, payload) => {
  const response = await api.post(`/api_db_agent/${agentType}/train`, payload, {
    headers: { 'accept': 'application/json', 'Content-Type': 'application/json' }
  });
  if (response.status !== 200) throw new Error('Ошибка запуска обучения');
  return response.data; // { agent, task_id }
};

export const evaluateModel = async (agentType, payload) => {
  const response = await api.post(`/api_db_agent/${agentType}/evaluate`, payload, {
    headers: { 'accept': 'application/json', 'Content-Type': 'application/json' }
  });
  if (response.status !== 200) throw new Error('Ошибка запуска оценки');
  return response.data; // { task_id }
};

export const getTaskStatus = async (taskId) => {
  const response = await api.get(`/api_db_agent/task_status/${taskId}`);
  if (response.status !== 200) throw new Error('Ошибка статуса задачи');
  return response.data;
};

export const promoteModel = async (agentId) => {
  const response = await api.post(`/api_db_agent/agents/${agentId}/promote`);
  if (response.status !== 200) throw new Error('Ошибка промо модели');
  return response.data;
};

export const getNewsBackground = async (coinId) => {
  const response = await api.get(`/api_db_agent/news/background/${coinId}`);
  return response.data;
};

export const recalcNewsBackground = async ({ coins = [] } = {}) => {
  const response = await api.post(`/api_db_agent/news/recalc_background`, null, { params: { coins } });
  if (response.status !== 200) throw new Error('Ошибка пересчета новостного фона');
  return response.data; // { task_id }
};
