import api from './api';

// Unified ML API service

export const listModels = async (type) => {
  const response = await api.get('/api_db_agent/models', { params: { type } });
  if (response.status !== 200) throw new Error('Ошибка загрузки моделей');
  return response.data;
};

export const trainModel = async (type, payload) => {
  const url = `/api_db_agent/${type}/train`;
  const response = await api.post(url, payload, {
    headers: { 'accept': 'application/json', 'Content-Type': 'application/json' }
  });
  if (response.status !== 200) throw new Error('Ошибка запуска обучения');
  return response.data; // { task_id }
};

export const evaluateModel = async (type, payload) => {
  const url = `/api_db_agent/${type}/evaluate`;
  const response = await api.post(url, payload, {
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

export const promoteModel = async (modelId) => {
  const response = await api.post(`/api_db_agent/models/${modelId}/promote`);
  if (response.status !== 200) throw new Error('Ошибка промо модели');
  return response.data;
};

// News specific
export const recalcNewsBackground = async ({ coins = '', window_hours = 24, decay_factor = 0.95, force_recalculate = false }) => {
  const response = await api.post('/api_db_agent/news/recalc_background', null, {
    params: { coins, window_hours, decay_factor, force_recalculate }
  });
  if (response.status !== 200) throw new Error('Ошибка запуска пересчета новостного фона');
  return response.data; // { task_id }
};

export const getNewsBackground = async ({ coin_id, start_time, end_time, limit = 1000 }) => {
  const response = await api.get(`/api_db_agent/news/background/${coin_id}`, {
    params: { start_time, end_time, limit }
  });
  if (response.status !== 200) throw new Error('Ошибка получения новостного фона');
  return response.data;
};

export const getNewsCoins = async () => {
  const response = await api.get('/api_db_agent/news/coins');
  if (response.status !== 200) throw new Error('Ошибка получения монет с новостями');
  return response.data;
};

export default {
  listModels,
  trainModel,
  evaluateModel,
  getTaskStatus,
  promoteModel,
  recalcNewsBackground,
  getNewsBackground,
  getNewsCoins,
  
  // Trade_time specific methods
  tradeTime: {
    train: (coinId, startDate, endDate, extraConfig) => 
      api.post(`/api_db_agent/trade_time/train`, { coin_id: coinId, start_date: startDate, end_date: endDate, extra_config: extraConfig }),
    evaluate: (coinId, startDate, endDate, modelPath, extraConfig) => 
      api.post(`/api_db_agent/trade_time/evaluate`, { coin_id: coinId, start_date: startDate, end_date: endDate, model_path: modelPath, extra_config: extraConfig }),
    getModels: (agentId) => api.get(`/api_db_agent/trade_time/models/${agentId}`),
    predict: (coinId, startDate, endDate, modelPath) => 
      api.post(`/api_db_agent/trade_time/predict`, { coin_id: coinId, start_date: startDate, end_date: endDate, model_path: modelPath })
  },

  // Risk specific methods
  risk: {
    train: (coinId, startDate, endDate, extraConfig) => 
      api.post(`/api_db_agent/risk/train`, { coin_id: coinId, start_date: startDate, end_date: endDate, extra_config: extraConfig }),
    evaluate: (coinId, startDate, endDate, modelPath, extraConfig) => 
      api.post(`/api_db_agent/risk/evaluate`, { coin_id: coinId, start_date: startDate, end_date: endDate, model_path: modelPath, extra_config: extraConfig }),
    getModels: (agentId) => api.get(`/api_db_agent/risk/models/${agentId}`),
    predict: (coinId, startDate, endDate, modelPath) => 
      api.post(`/api_db_agent/risk/predict`, { coin_id: coinId, start_date: startDate, end_date: endDate, model_path: modelPath })
  }
};

// Note: duplicate legacy exports removed
