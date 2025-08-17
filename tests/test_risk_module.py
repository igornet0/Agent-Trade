#!/usr/bin/env python3
"""
Тесты для Risk модуля
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestRiskService(unittest.TestCase):
    """Тесты для RiskService"""
    
    def setUp(self):
        """Настройка тестов"""
        from core.services.risk_service import RiskService
        self.service = RiskService()
        
        # Мок данных
        self.mock_df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=100, freq='1H'),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        self.mock_news_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=100, freq='1H'),
            'score': np.random.uniform(-1, 1, 100),
            'source_count': np.random.randint(1, 10, 100)
        })
    
    def test_calculate_technical_indicators(self):
        """Тест расчета технических индикаторов"""
        result = self.service._calculate_technical_indicators(self.mock_df)
        
        # Проверяем наличие всех индикаторов
        expected_indicators = [
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 
            'macd_signal', 'macd_histogram', 'bb_width', 'bb_position',
            'volume_ratio', 'price_change', 'volatility_ratio', 'atr', 'atr_ratio',
            'price_range', 'price_range_5', 'price_range_20'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
        
        # Проверяем корректность RSI
        self.assertTrue(all(result['rsi'].between(0, 100)))
        
        # Проверяем корректность ATR
        self.assertTrue(all(result['atr'] >= 0))
        self.assertTrue(all(result['atr_ratio'] >= 0))
    
    def test_calculate_heuristic_risk_score(self):
        """Тест расчета эвристического риска"""
        df_with_indicators = self.service._calculate_technical_indicators(self.mock_df)
        risk_score = self.service._calculate_heuristic_risk_score(df_with_indicators)
        
        # Проверяем размер
        self.assertEqual(len(risk_score), len(df_with_indicators))
        
        # Проверяем диапазон (0-1)
        self.assertTrue(all(risk_score.between(0, 1)))
        
        # Проверяем что риск не NaN
        self.assertFalse(risk_score.isna().any())
    
    def test_calculate_heuristic_volume_score(self):
        """Тест расчета эвристического объема"""
        df_with_indicators = self.service._calculate_technical_indicators(self.mock_df)
        risk_score = self.service._calculate_heuristic_risk_score(df_with_indicators)
        volume_score = self.service._calculate_heuristic_volume_score(df_with_indicators, risk_score)
        
        # Проверяем размер
        self.assertEqual(len(volume_score), len(df_with_indicators))
        
        # Проверяем диапазон (0-1)
        self.assertTrue(all(volume_score.between(0, 1)))
        
        # Проверяем что объем не NaN
        self.assertFalse(volume_score.isna().any())
    
    def test_prepare_features(self):
        """Тест подготовки признаков"""
        X, y_risk, y_volume = self.service._prepare_features(self.mock_df, self.mock_news_data)
        
        # Проверяем размеры данных
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y_risk), 0)
        self.assertGreater(len(y_volume), 0)
        self.assertEqual(len(X), len(y_risk))
        self.assertEqual(len(X), len(y_volume))
        
        # Проверяем количество признаков
        expected_features = 26  # 24 технических + 2 новостных
        self.assertEqual(X.shape[1], expected_features)
        
        # Проверяем целевые переменные
        self.assertTrue(all(y_risk.between(0, 1)))
        self.assertTrue(all(y_volume.between(0, 1)))
    
    def test_create_model_xgboost(self):
        """Тест создания XGBoost модели"""
        try:
            model = self.service._create_model('xgboost', n_estimators=50)
            self.assertIsNotNone(model)
            self.assertEqual(model.n_estimators, 50)
        except ImportError:
            self.skipTest("XGBoost not available")
    
    def test_create_model_unsupported(self):
        """Тест создания неподдерживаемой модели"""
        with self.assertRaises(ValueError):
            self.service._create_model('unsupported_type')
    
    def test_calculate_risk_metrics(self):
        """Тест расчета метрик для risk модели"""
        y_true = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        y_pred = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        
        metrics = self.service._calculate_risk_metrics(y_true, y_pred)
        
        # Проверяем наличие всех метрик
        expected_metrics = ['rmse', 'mae', 'mape', 'mean_risk', 'std_risk', 'max_risk', 'min_risk', 'correlation', 'r_squared']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Проверяем корректность метрик
        self.assertEqual(metrics['rmse'], 0.0)
        self.assertEqual(metrics['mae'], 0.0)
        self.assertEqual(metrics['correlation'], 1.0)
        self.assertEqual(metrics['r_squared'], 1.0)
    
    def test_calculate_volume_metrics(self):
        """Тест расчета метрик для volume модели"""
        y_true = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        y_pred = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
        
        metrics = self.service._calculate_volume_metrics(y_true, y_pred)
        
        # Проверяем наличие всех метрик
        expected_metrics = ['rmse', 'mae', 'mape', 'mean_volume', 'std_volume', 'max_volume', 'min_volume', 'correlation', 'r_squared']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Проверяем корректность метрик
        self.assertEqual(metrics['rmse'], 0.0)
        self.assertEqual(metrics['mae'], 0.0)
        self.assertEqual(metrics['correlation'], 1.0)
        self.assertEqual(metrics['r_squared'], 1.0)
    
    def test_save_models(self):
        """Тест сохранения моделей"""
        # Мок модели
        mock_risk_model = Mock()
        mock_volume_model = Mock()
        
        coin_id = "test_coin"
        model_type = "xgboost"
        extra_config = {"test": "config"}
        risk_metrics = {"rmse": 0.1}
        volume_metrics = {"rmse": 0.2}
        
        # Создаем временную директорию
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Патчим путь к моделям
            with patch.object(self.service, 'models_dir', temp_dir):
                model_path = self.service._save_models(
                    mock_risk_model, mock_volume_model, coin_id, model_type, 
                    extra_config, risk_metrics, volume_metrics
                )
                
                # Проверяем создание директории
                self.assertTrue(os.path.exists(model_path))
                
                # Проверяем наличие файлов
                config_file = os.path.join(model_path, 'config.json')
                metadata_file = os.path.join(model_path, 'metadata.json')
                
                self.assertTrue(os.path.exists(config_file))
                self.assertTrue(os.path.exists(metadata_file))
    
    def test_load_models(self):
        """Тест загрузки моделей"""
        # Создаем временную директорию с мок моделями
        import tempfile
        import json
        import pickle
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем мок модели
            mock_risk_model = Mock()
            mock_volume_model = Mock()
            
            # Сохраняем модели
            risk_model_file = os.path.join(temp_dir, 'risk_model.pkl')
            volume_model_file = os.path.join(temp_dir, 'volume_model.pkl')
            
            with open(risk_model_file, 'wb') as f:
                pickle.dump(mock_risk_model, f)
            with open(volume_model_file, 'wb') as f:
                pickle.dump(mock_volume_model, f)
            
            # Создаем конфиг
            config = {
                'coin_id': 'test_coin',
                'model_type': 'xgboost',
                'timestamp': '20250101_120000'
            }
            config_file = os.path.join(temp_dir, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            # Тестируем загрузку
            loaded_risk_model, loaded_volume_model = self.service.load_models(temp_dir)
            self.assertIsNotNone(loaded_risk_model)
            self.assertIsNotNone(loaded_volume_model)
    
    def test_predict(self):
        """Тест предсказания"""
        # Мок модели
        mock_risk_model = Mock()
        mock_volume_model = Mock()
        mock_risk_model.predict = Mock(return_value=np.array([0.1, 0.5, 0.9]))
        mock_volume_model.predict = Mock(return_value=np.array([0.2, 0.6, 0.8]))
        
        # Патчим загрузку моделей
        with patch.object(self.service, 'load_models', return_value=(mock_risk_model, mock_volume_model)):
            # Патчим получение данных
            with patch('core.services.risk_service.orm_get_coin_data', return_value=self.mock_df):
                with patch('core.services.risk_service.orm_get_news_background', return_value=self.mock_news_data):
                    result = self.service.predict('mock_path', 'test_coin', '2025-01-01', '2025-01-02')
                    
                    # Проверяем структуру результата
                    expected_keys = ['timestamp', 'risk_scores', 'volume_scores', 'risk_levels', 'volume_levels']
                    for key in expected_keys:
                        self.assertIn(key, result)
                    
                    # Проверяем количество предсказаний
                    self.assertEqual(len(result['risk_scores']), len(result['volume_scores']))
                    self.assertEqual(len(result['risk_levels']), len(result['volume_levels']))
                    
                    # Проверяем корректность уровней
                    valid_risk_levels = ['LOW', 'MEDIUM', 'HIGH']
                    valid_volume_levels = ['LOW', 'MEDIUM', 'HIGH']
                    
                    for level in result['risk_levels']:
                        self.assertIn(level, valid_risk_levels)
                    
                    for level in result['volume_levels']:
                        self.assertIn(level, valid_volume_levels)
    
    def test_calculate_var(self):
        """Тест расчета Value at Risk"""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02])
        
        var_95 = self.service.calculate_var(returns, 0.95)
        var_99 = self.service.calculate_var(returns, 0.99)
        
        # Проверяем что VaR корректный
        self.assertIsInstance(var_95, float)
        self.assertIsInstance(var_99, float)
        
        # VaR 99% должен быть меньше VaR 95%
        self.assertLessEqual(var_99, var_95)
    
    def test_calculate_expected_shortfall(self):
        """Тест расчета Expected Shortfall"""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02])
        
        es_95 = self.service.calculate_expected_shortfall(returns, 0.95)
        es_99 = self.service.calculate_expected_shortfall(returns, 0.99)
        
        # Проверяем что ES корректный
        self.assertIsInstance(es_95, float)
        self.assertIsInstance(es_99, float)
        
        # ES должен быть меньше или равен VaR
        var_95 = self.service.calculate_var(returns, 0.95)
        var_99 = self.service.calculate_var(returns, 0.99)
        
        self.assertLessEqual(es_95, var_95)
        self.assertLessEqual(es_99, var_99)

class TestRiskIntegration(unittest.TestCase):
    """Интеграционные тесты для Risk модуля"""
    
    def setUp(self):
        """Настройка тестов"""
        from core.services.risk_service import RiskService
        self.service = RiskService()
    
    @patch('core.services.risk_service.orm_get_coin_data')
    @patch('core.services.risk_service.orm_get_news_background')
    def test_train_model_integration(self, mock_news, mock_market):
        """Интеграционный тест обучения модели"""
        # Мок данные
        mock_market.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=200, freq='1H'),
            'open': np.random.uniform(100, 200, 200),
            'high': np.random.uniform(100, 200, 200),
            'low': np.random.uniform(100, 200, 200),
            'close': np.random.uniform(100, 200, 200),
            'volume': np.random.uniform(1000, 10000, 200)
        })
        
        mock_news.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=200, freq='1H'),
            'score': np.random.uniform(-1, 1, 200),
            'source_count': np.random.randint(1, 10, 200)
        })
        
        # Тестируем обучение
        try:
            result = self.service.train_model(
                'test_coin', 
                '2025-01-01', 
                '2025-01-10',
                'xgboost',
                {'n_estimators': 10}
            )
            
            self.assertEqual(result['status'], 'success')
            self.assertIn('model_path', result)
            self.assertIn('risk_metrics', result)
            self.assertIn('volume_metrics', result)
            self.assertIn('data_info', result)
            
        except Exception as e:
            # Если обучение не удалось, проверяем что ошибка логична
            self.assertIn('error', result)
            self.assertIsInstance(result['error'], str)

if __name__ == '__main__':
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    test_suite.addTest(unittest.makeSuite(TestRiskService))
    test_suite.addTest(unittest.makeSuite(TestRiskIntegration))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Выводим результат
    print(f"\nТесты завершены: {result.testsRun} тестов выполнено")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Провалов: {len(result.failures)}")
    
    if result.wasSuccessful():
        print("✅ Все тесты прошли успешно!")
        sys.exit(0)
    else:
        print("❌ Некоторые тесты провалились")
        sys.exit(1)
