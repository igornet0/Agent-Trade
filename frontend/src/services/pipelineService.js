import api from './api';

// Pipeline management
export const savePipeline = async (pipelineData) => {
  const response = await api.post('/pipeline/save', pipelineData);
  if (response.status !== 200) throw new Error('Ошибка сохранения пайплайна');
  return response.data;
};

export const loadPipeline = async (pipelineId) => {
  const response = await api.get(`/pipeline/${pipelineId}`);
  if (response.status !== 200) throw new Error('Ошибка загрузки пайплайна');
  return response.data;
};

export const listPipelines = async () => {
  const response = await api.get('/pipeline/');
  if (response.status !== 200) throw new Error('Ошибка загрузки пайплайнов');
  return response.data;
};

export const deletePipeline = async (pipelineId) => {
  const response = await api.delete(`/pipeline/${pipelineId}`);
  if (response.status !== 200) throw new Error('Ошибка удаления пайплайна');
  return response.data;
};

// Pipeline execution
export const runPipeline = async (pipelineId, config) => {
  const payload = pipelineId ? { pipeline_id: pipelineId } : config;
  const response = await api.post('/pipeline/run', payload);
  if (response.status !== 200) throw new Error('Ошибка запуска пайплайна');
  return response.data; // { task_id }
};

export const runPipelineBacktest = async (pipelineId, backtestConfig) => {
  const payload = {
    pipeline_id: pipelineId,
    ...backtestConfig
  };
  const response = await api.post('/pipeline/run', payload);
  if (response.status !== 200) throw new Error('Ошибка запуска бэктеста');
  return response.data; // { task_id }
};

export const revokePipeline = async (taskId) => {
  const response = await api.post(`/pipeline/tasks/${taskId}/revoke`);
  if (response.status !== 200) throw new Error('Ошибка отмены пайплайна');
  return response.data;
};

// Backtest management
export const getBacktests = async () => {
  const response = await api.get('/pipeline/backtests');
  if (response.status !== 200) throw new Error('Ошибка загрузки бэктестов');
  return response.data;
};

export const getBacktest = async (backtestId) => {
  const response = await api.get(`/pipeline/backtests/${backtestId}`);
  if (response.status !== 200) throw new Error('Ошибка загрузки бэктеста');
  return response.data;
};

export const deleteBacktest = async (backtestId) => {
  const response = await api.delete(`/pipeline/backtests/${backtestId}`);
  if (response.status !== 200) throw new Error('Ошибка удаления бэктеста');
  return response.data;
};

// Artifacts
export const downloadArtifact = async (artifactPath) => {
  const response = await api.get(`/pipeline/artifacts/${artifactPath}`, {
    responseType: 'blob'
  });
  if (response.status !== 200) throw new Error('Ошибка загрузки артефакта');
  return response.data;
};

export const listArtifacts = async (backtestId) => {
  const response = await api.get(`/pipeline/backtests/${backtestId}/artifacts`);
  if (response.status !== 200) throw new Error('Ошибка загрузки списка артефактов');
  return response.data;
};

// Pipeline templates
export const getPipelineTemplates = async () => {
  const response = await api.get('/pipeline/templates');
  if (response.status !== 200) throw new Error('Ошибка загрузки шаблонов');
  return response.data;
};

export const savePipelineTemplate = async (templateData) => {
  const response = await api.post('/pipeline/templates', templateData);
  if (response.status !== 200) throw new Error('Ошибка сохранения шаблона');
  return response.data;
};

export default {
  // Pipeline management
  savePipeline,
  loadPipeline,
  listPipelines,
  deletePipeline,
  
  // Pipeline execution
  runPipeline,
  runPipelineBacktest,
  revokePipeline,
  
  // Backtest management
  getBacktests,
  getBacktest,
  deleteBacktest,
  
  // Artifacts
  downloadArtifact,
  listArtifacts,
  
  // Templates
  getPipelineTemplates,
  savePipelineTemplate,
};


