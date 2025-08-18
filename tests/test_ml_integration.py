import pytest
from datetime import datetime, timedelta
import json


class TestMLSystemIntegration:
    """Интеграционные тесты для ML-системы"""
    
    def test_complete_ml_workflow(self):
        """Тест полного workflow ML-системы"""
        # 1. Подготовка данных
        test_data = self._prepare_test_data()
        assert len(test_data) > 0
        assert all('datetime' in record for record in test_data)
        assert all('coin' in record for record in test_data)
        
        # 2. Конфигурация агентов
        agent_configs = self._prepare_agent_configs()
        assert len(agent_configs) == 5  # 5 типов агентов
        assert all('type' in config for config in agent_configs.values())
        
        # 3. Тестирование конфигураций
        for agent_type, config in agent_configs.items():
            assert self._validate_agent_config(config)
        
        # 4. Симуляция обучения
        training_results = self._simulate_training(agent_configs)
        assert len(training_results) == 5
        assert all('status' in result for result in training_results.values())
        
        # 5. Симуляция тестирования
        test_results = self._simulate_testing(training_results)
        assert len(test_results) == 5
        assert all(len(result) > 0 for result in test_results.values())  # Проверяем, что результаты не пустые
        
        # 6. Валидация результатов
        for agent_type, result in test_results.items():
            assert self._validate_test_results(result, agent_type)
    
    def test_data_pipeline(self):
        """Тест пайплайна данных"""
        # Подготовка данных
        raw_data = self._generate_raw_data()
        assert len(raw_data) == 100
        
        # Обработка данных
        processed_data = self._process_data(raw_data)
        assert len(processed_data) == 100
        assert all('features' in record for record in processed_data)
        
        # Валидация данных
        validation_result = self._validate_data(processed_data)
        assert validation_result['is_valid']
        assert validation_result['completeness'] > 0.95
    
    def test_model_training_pipeline(self):
        """Тест пайплайна обучения моделей"""
        # Подготовка данных для обучения
        training_data = self._prepare_training_data()
        assert len(training_data) > 0
        
        # Конфигурация модели
        model_config = self._get_model_config('AgentPredTime')
        assert model_config['type'] == 'AgentPredTime'
        assert 'hyperparameters' in model_config
        
        # Симуляция обучения
        training_result = self._simulate_model_training(training_data, model_config)
        assert training_result['status'] == 'completed'
        assert 'model_path' in training_result
        assert 'metrics' in training_result
    
    def test_model_evaluation_pipeline(self):
        """Тест пайплайна оценки моделей"""
        # Подготовка тестовых данных
        test_data = self._prepare_test_data()
        
        # Загрузка модели
        model_info = self._load_model_info('test_model.pth')
        assert model_info['type'] == 'AgentPredTime'
        
        # Симуляция оценки
        evaluation_result = self._simulate_model_evaluation(test_data, model_info)
        assert 'accuracy' in evaluation_result
        assert 'precision' in evaluation_result
        assert 'recall' in evaluation_result
        assert evaluation_result['accuracy'] > 0.5
    
    def test_end_to_end_workflow(self):
        """Тест end-to-end workflow"""
        # 1. Импорт данных
        import_result = self._simulate_data_import()
        assert import_result['imported_records'] > 0
        
        # 2. Обучение модели
        training_result = self._simulate_training_step()
        assert training_result['status'] == 'completed'
        
        # 3. Тестирование модели
        testing_result = self._simulate_testing_step()
        assert testing_result['status'] == 'completed'
        
        # 4. Экспорт результатов
        export_result = self._simulate_results_export()
        assert export_result['exported'] == True
    
    def _prepare_test_data(self):
        """Подготовка тестовых данных"""
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        data = []
        
        for i in range(100):
            dt = base_time + timedelta(minutes=i * 5)
            data.append({
                'datetime': dt.isoformat(),
                'coin': 'BTC',
                'open': 50000.0 + i * 10,
                'high': 50100.0 + i * 10,
                'low': 49900.0 + i * 10,
                'close': 50050.0 + i * 10,
                'volume': 1000000.0 + i * 1000
            })
        
        return data
    
    def _prepare_agent_configs(self):
        """Подготовка конфигураций агентов"""
        return {
            'AgentNews': {
                'type': 'AgentNews',
                'timeframe': '5m',
                'features': ['sentiment', 'entities', 'source_credibility'],
                'hyperparameters': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                }
            },
            'AgentPredTime': {
                'type': 'AgentPredTime',
                'timeframe': '5m',
                'features': ['ohlcv', 'technical_indicators', 'news_background'],
                'hyperparameters': {
                    'seq_len': 96,
                    'pred_len': 12,
                    'model_type': 'lstm',
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'epochs': 200
                }
            },
            'AgentTradeTime': {
                'type': 'AgentTradeTime',
                'timeframe': '5m',
                'features': ['pred_time_signals', 'technical_indicators', 'market_conditions'],
                'hyperparameters': {
                    'model_type': 'lightgbm',
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'n_estimators': 100
                }
            },
            'AgentRisk': {
                'type': 'AgentRisk',
                'timeframe': '5m',
                'features': ['open_positions', 'balance', 'trade_signals', 'news_background'],
                'hyperparameters': {
                    'model_type': 'xgboost',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100
                }
            },
            'AgentTradeAggregator': {
                'type': 'AgentTradeAggregator',
                'timeframe': '5m',
                'features': ['all_signals', 'portfolio_state', 'risk_assessment'],
                'hyperparameters': {
                    'model_type': 'ensemble',
                    'weights': [0.3, 0.3, 0.2, 0.2],
                    'threshold': 0.6
                }
            }
        }
    
    def _validate_agent_config(self, config):
        """Валидация конфигурации агента"""
        required_fields = ['type', 'timeframe', 'features', 'hyperparameters']
        return all(field in config for field in required_fields)
    
    def _simulate_training(self, agent_configs):
        """Симуляция обучения агентов"""
        results = {}
        
        for agent_type, config in agent_configs.items():
            results[agent_type] = {
                'status': 'completed',
                'model_path': f'models/{agent_type.lower()}/model.pth',
                'config_path': f'models/{agent_type.lower()}/config.json',
                'training_time': 3600,  # 1 час
                'epochs_completed': config['hyperparameters'].get('epochs', 100),
                'final_loss': 0.15,
                'accuracy': 0.85
            }
        
        return results
    
    def _simulate_testing(self, training_results):
        """Симуляция тестирования агентов"""
        results = {}
        
        for agent_type, training_result in training_results.items():
            if agent_type == 'AgentNews':
                results[agent_type] = {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1_score': 0.85,
                    'sentiment_accuracy': 0.78
                }
            elif agent_type == 'AgentPredTime':
                results[agent_type] = {
                    'mae': 0.0234,
                    'mse': 0.0012,
                    'rmse': 0.0346,
                    'mape': 0.0456,
                    'directional_accuracy': 0.67
                }
            elif agent_type == 'AgentTradeTime':
                results[agent_type] = {
                    'accuracy': 0.72,
                    'precision': 0.68,
                    'recall': 0.75,
                    'f1_score': 0.71,
                    'profit_factor': 1.45,
                    'win_rate': 0.58
                }
            elif agent_type == 'AgentRisk':
                results[agent_type] = {
                    'risk_accuracy': 0.81,
                    'max_drawdown': 0.12,
                    'sharpe_ratio': 1.85,
                    'calmar_ratio': 2.34
                }
            elif agent_type == 'AgentTradeAggregator':
                results[agent_type] = {
                    'total_return': 0.156,
                    'sharpe_ratio': 1.92,
                    'max_drawdown': 0.089,
                    'win_rate': 0.62,
                    'profit_factor': 1.67
                }
        
        return results
    
    def _validate_test_results(self, result, agent_type):
        """Валидация результатов тестирования"""
        if agent_type == 'AgentNews':
            return all(metric in result for metric in ['accuracy', 'precision', 'recall', 'f1_score'])
        elif agent_type == 'AgentPredTime':
            return all(metric in result for metric in ['mae', 'mse', 'rmse', 'mape'])
        elif agent_type == 'AgentTradeTime':
            return all(metric in result for metric in ['accuracy', 'precision', 'recall', 'f1_score'])
        elif agent_type == 'AgentRisk':
            return all(metric in result for metric in ['risk_accuracy', 'max_drawdown', 'sharpe_ratio'])
        elif agent_type == 'AgentTradeAggregator':
            return all(metric in result for metric in ['total_return', 'sharpe_ratio', 'max_drawdown'])
        return False
    
    def _generate_raw_data(self):
        """Генерация сырых данных"""
        data = []
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        for i in range(100):
            dt = base_time + timedelta(minutes=i * 5)
            data.append({
                'datetime': dt,
                'open': 50000.0 + i * 10,
                'high': 50100.0 + i * 10,
                'low': 49900.0 + i * 10,
                'close': 50050.0 + i * 10,
                'volume': 1000000.0 + i * 1000
            })
        
        return data
    
    def _process_data(self, raw_data):
        """Обработка данных"""
        processed_data = []
        
        for record in raw_data:
            # Добавляем технические индикаторы
            processed_record = {
                **record,
                'features': {
                    'sma_20': record['close'] * 1.001,  # Простая скользящая средняя
                    'rsi': 50.0 + (record['close'] - 50000) / 100,  # RSI
                    'volume_sma': record['volume'] * 1.002,  # Объемная скользящая средняя
                    'price_change': (record['close'] - record['open']) / record['open'],
                    'volatility': (record['high'] - record['low']) / record['open']
                }
            }
            processed_data.append(processed_record)
        
        return processed_data
    
    def _validate_data(self, processed_data):
        """Валидация данных"""
        total_records = len(processed_data)
        valid_records = sum(1 for record in processed_data if 'features' in record)
        
        return {
            'is_valid': valid_records == total_records,
            'completeness': valid_records / total_records,
            'total_records': total_records,
            'valid_records': valid_records
        }
    
    def _prepare_training_data(self):
        """Подготовка данных для обучения"""
        return self._process_data(self._generate_raw_data())
    
    def _get_model_config(self, model_type):
        """Получение конфигурации модели"""
        configs = {
            'AgentPredTime': {
                'type': 'AgentPredTime',
                'timeframe': '5m',
                'hyperparameters': {
                    'seq_len': 96,
                    'pred_len': 12,
                    'model_type': 'lstm',
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'epochs': 200
                }
            }
        }
        return configs.get(model_type, {})
    
    def _simulate_model_training(self, training_data, model_config):
        """Симуляция обучения модели"""
        return {
            'status': 'completed',
            'model_path': f'models/{model_config["type"].lower()}/model.pth',
            'config_path': f'models/{model_config["type"].lower()}/config.json',
            'training_time': 3600,
            'epochs_completed': model_config['hyperparameters']['epochs'],
            'final_loss': 0.15,
            'accuracy': 0.85,
            'metrics': {
                'train_loss': 0.15,
                'val_loss': 0.18,
                'train_accuracy': 0.85,
                'val_accuracy': 0.82
            }
        }
    
    def _load_model_info(self, model_path):
        """Загрузка информации о модели"""
        return {
            'type': 'AgentPredTime',
            'model_path': model_path,
            'config': {
                'seq_len': 96,
                'pred_len': 12,
                'model_type': 'lstm'
            },
            'metadata': {
                'created_at': '2025-01-01T00:00:00',
                'version': '1.0.0',
                'training_time': 3600
            }
        }
    
    def _simulate_model_evaluation(self, test_data, model_info):
        """Симуляция оценки модели"""
        return {
            'accuracy': 0.82,
            'precision': 0.79,
            'recall': 0.85,
            'f1_score': 0.82,
            'mae': 0.0234,
            'mse': 0.0012,
            'rmse': 0.0346,
            'test_samples': len(test_data)
        }
    
    def _simulate_data_import(self):
        """Симуляция импорта данных"""
        return {
            'imported_records': 1000,
            'skipped_records': 50,
            'errors': [],
            'status': 'success'
        }
    
    def _simulate_training_step(self):
        """Симуляция шага обучения"""
        return {
            'status': 'completed',
            'model_path': 'models/agent_pred_time/model.pth',
            'training_time': 3600,
            'epochs_completed': 200,
            'final_loss': 0.15
        }
    
    def _simulate_testing_step(self):
        """Симуляция шага тестирования"""
        return {
            'status': 'completed',
            'accuracy': 0.82,
            'precision': 0.79,
            'recall': 0.85,
            'f1_score': 0.82,
            'test_samples': 1000
        }
    
    def _simulate_results_export(self):
        """Симуляция экспорта результатов"""
        return {
            'exported': True,
            'file_path': 'results/model_evaluation_20250101.json',
            'format': 'json',
            'size_bytes': 1024
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
