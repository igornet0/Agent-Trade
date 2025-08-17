import api from './api';

export const savePipeline = async (config) => {
  const response = await api.post('/pipeline/save', config, {
    headers: { 'accept': 'application/json', 'Content-Type': 'application/json' }
  });
  if (response.status !== 200) throw new Error('Ошибка сохранения пайплайна');
  return response.data;
};

export const runPipeline = async (config) => {
  const response = await api.post('/pipeline/run', config, {
    headers: { 'accept': 'application/json', 'Content-Type': 'application/json' }
  });
  if (response.status !== 200) throw new Error('Ошибка запуска пайплайна');
  return response.data; // { task_id }
};

export const loadPipeline = async (pipelineId) => {
  const response = await api.get(`/pipeline/${pipelineId}`);
  if (response.status !== 200) throw new Error('Пайплайн не найден');
  return response.data;
};


