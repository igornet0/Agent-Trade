import api from './api';

export const get_coins = async (coinsData) => {
  const response = await api.get('/coins/get_coins/', coinsData);

  if (response.status !== 200) throw new Error('Ошибка загрузки монет');

  return response.data;
};

export const get_coin_time_line = async (coinsDataTimeLine) => {
  const params = {
    coin_id: coinsDataTimeLine.coin_id,
    timeframe: coinsDataTimeLine.timeframe,
    size_page: coinsDataTimeLine.size_page || 100  // Добавляем размер страницы
  };

  if (coinsDataTimeLine.last_timestamp) {
    params.last_timestamp = coinsDataTimeLine.last_timestamp;
  }
  
  const response = await api.get('/coins/get_coin/', {
    params: params, 
    headers: { 'Content-Type': 'application/json' }
  });

  if (response.status !== 200) throw new Error('Ошибка загрузки монет');

  return response.data;
};

export const get_agents = async (status=null, type=null) => {
  const response = await api.get('/api_db_agent/agents/', {
    params: {
      status: status,
      type: type
    }
  });

  if (response.status !== 200) throw new Error('Ошибка загрузки агентов');

  return response.data;
};

export const get_agent_types = async () => {
  const response = await api.get('/api_db_agent/agents_types/');

  if (response.status !== 200) throw new Error('Ошибка загрузки агентов');

  return response.data;
}

export const get_available_features = async () => {
  const response = await api.get('/api_db_agent/available_features/');

  if (response.status !== 200) throw new Error('Ошибка загрузки агентов');

  return response.data;
}

export const train_new_agent = async (agentData) => {
  console.log(agentData);
  const response = await api.post('/api_db_agent/train_new_agent/', agentData,{
          headers: {
            'accept': 'application/json',
            // 'Authorization': `Bearer ${localStorage.getItem('access_token')}` ,
            'Content-Type': 'application/json'
          }
      });
  
  if (response.status !== 200) throw new Error('Ошибка обучения агента');

  return response.data;
}

export const delete_agent = async (agentId) => {
  const response = await api.post(`/api_db_agent/delete_agent/${agentId}/`, {
    headers: {
      // 'accept': 'application/json',
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
      // 'Content-Type': 'application/json'
    }
  });
  if (response.status !== 200) throw new Error('Ошибка удаления агента');
  return response.data;
}

export const evaluate_agent = async ({ agent_id, coins, timeframe = '5m', start = null, end = null }) => {
  const payload = { agent_id, coins, timeframe, start, end };
  const response = await api.post('/api_db_agent/evaluate', payload, {
    headers: {
      'accept': 'application/json',
      'Content-Type': 'application/json'
    }
  });
  if (response.status !== 200) throw new Error('Ошибка запуска оценки модели');
  return response.data; // { task_id }
}

export const get_task_status = async (taskId) => {
  const response = await api.get(`/api_db_agent/task_status/${taskId}`);
  if (response.status !== 200) throw new Error('Ошибка статуса задачи');
  return response.data;
}