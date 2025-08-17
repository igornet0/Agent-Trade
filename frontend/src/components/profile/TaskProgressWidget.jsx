import React, { useState, useEffect } from 'react';
import { getTaskStatus } from '../../services/mlService';

const TaskProgressWidget = ({ taskId, onComplete, onError, autoStart = true }) => {
  const [task, setTask] = useState(null);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (taskId && autoStart) {
      startPolling();
    }
    
    return () => {
      // Cleanup polling on unmount
      if (polling) {
        setPolling(false);
      }
    };
  }, [taskId, autoStart]);

  const startPolling = () => {
    if (!taskId || polling) return;
    
    setPolling(true);
    setError(null);
    
    const pollInterval = setInterval(async () => {
      try {
        const taskData = await getTaskStatus(taskId);
        setTask(taskData);
        
        if (taskData.ready) {
          clearInterval(pollInterval);
          setPolling(false);
          
          if (taskData.successful) {
            onComplete?.(taskData);
          } else {
            onError?.(taskData);
          }
        }
      } catch (err) {
        console.error(`Error polling task ${taskId}:`, err);
        setError(err.message || 'Ошибка получения статуса задачи');
        clearInterval(pollInterval);
        setPolling(false);
        onError?.({ error: err.message });
      }
    }, 2000);
  };

  const stopPolling = () => {
    setPolling(false);
  };

  const getStatusIcon = (state) => {
    switch (state?.toLowerCase()) {
      case 'pending':
        return <div className="w-4 h-4 bg-yellow-500 rounded-full animate-pulse"></div>;
      case 'started':
        return <div className="w-4 h-4 bg-blue-500 rounded-full animate-pulse"></div>;
      case 'progress':
        return <div className="w-4 h-4 bg-blue-500 rounded-full animate-pulse"></div>;
      case 'success':
        return <div className="w-4 h-4 bg-green-500 rounded-full"></div>;
      case 'failure':
        return <div className="w-4 h-4 bg-red-500 rounded-full"></div>;
      default:
        return <div className="w-4 h-4 bg-gray-400 rounded-full"></div>;
    }
  };

  const getStatusColor = (state) => {
    switch (state?.toLowerCase()) {
      case 'pending':
        return 'text-yellow-700 bg-yellow-100';
      case 'started':
        return 'text-blue-700 bg-blue-100';
      case 'progress':
        return 'text-blue-700 bg-blue-100';
      case 'success':
        return 'text-green-700 bg-green-100';
      case 'failure':
        return 'text-red-700 bg-red-100';
      default:
        return 'text-gray-700 bg-gray-100';
    }
  };

  if (!taskId) {
    return null;
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          {getStatusIcon(task?.state)}
          <span className="text-sm font-medium text-gray-700">
            Задача {taskId}
          </span>
        </div>
        
        <div className="flex items-center space-x-2">
          {task?.state && (
            <span className={`px-2 py-1 text-xs rounded-full font-medium ${getStatusColor(task.state)}`}>
              {task.state}
            </span>
          )}
          
          {polling && (
            <button
              onClick={stopPolling}
              className="text-xs text-gray-500 hover:text-gray-700"
            >
              Остановить
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {task?.meta?.progress !== undefined && (
        <div className="mb-3">
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span>Прогресс</span>
            <span>{task.meta.progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${task.meta.progress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Current Step */}
      {task?.meta?.current_step && (
        <div className="mb-3">
          <div className="text-xs text-gray-600 mb-1">Текущий шаг</div>
          <div className="text-sm text-gray-800 font-medium">
            {task.meta.current_step}
          </div>
        </div>
      )}

      {/* Logs */}
      {task?.meta?.logs && task.meta.logs.length > 0 && (
        <div className="mb-3">
          <div className="text-xs text-gray-600 mb-1">Логи</div>
          <div className="max-h-32 overflow-y-auto bg-gray-50 rounded p-2">
            {task.meta.logs.map((log, index) => (
              <div key={index} className="text-xs text-gray-700 font-mono">
                {log}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Task Result */}
      {task?.ready && task?.meta?.result && (
        <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded">
          <div className="text-sm font-medium text-green-800 mb-2">Результат выполнения:</div>
          <div className="text-sm text-green-700">
            {typeof task.meta.result === 'object' ? (
              <pre className="whitespace-pre-wrap text-xs">
                {JSON.stringify(task.meta.result, null, 2)}
              </pre>
            ) : (
              String(task.meta.result)
            )}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-end space-x-2 mt-3 pt-3 border-t border-gray-200">
        {!polling && !task?.ready && (
          <button
            onClick={startPolling}
            className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            Возобновить
          </button>
        )}
        
        {task?.ready && (
          <button
            onClick={() => onComplete?.(task)}
            className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
          >
            Просмотреть результат
          </button>
        )}
      </div>
    </div>
  );
};

export default TaskProgressWidget;
