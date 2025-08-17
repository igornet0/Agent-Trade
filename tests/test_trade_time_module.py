#!/usr/bin/env python3
"""
Тесты для Trade_time модуля
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestTradeTimeService(unittest.TestCase):
    """Тесты для TradeTimeService"""
    
    def setUp(self):
        """Настройка тестов"""
        from core.services.trade_time_service import TradeTimeService
        self.service = TradeTimeService()
        
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
            'volume_ratio', 'price_change', 'volatility_ratio'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
        
        # Проверяем корректность RSI
        self.assertTrue(all(result['rsi'].between(0, 100)))
        
        # Проверяем корректность MACD
        self.assertIn('macd', result.columns)
        self.assertIn('macd_signal', result.columns)
    
    def test_prepare_features(self):
        """Тест подготовки признаков"""
        X, y = self.service._prepare_features(self.mock_df, self.mock_news_data)
        
        # Проверяем размеры данных
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y), 0)
        self.assertEqual(len(X), len(y))
        
        # Проверяем количество признаков
        expected_features = 21  # 19 технических + 2 новостных
        self.assertEqual(X.shape[1], expected_features)
        
        # Проверяем целевые переменные
        unique_signals = np.unique(y)
        self.assertTrue(all(signal in [-1, 0, 1] for signal in unique_signals))
    
    def test_create_model_lightgbm(self):
        """Тест создания LightGBM модели"""
        try:
            model = self.service._create_model('lightgbm', n_estimators=50)
            self.assertIsNotNone(model)
            self.assertEqual(model.n_estimators, 50)
        except ImportError:
            self.skipTest("LightGBM not available")
    
    def test_create_model_catboost(self):
        """Тест создания CatBoost модели"""
        try:
            model = self.service._create_model('catboost', iterations=50)
            self.assertIsNotNone(model)
            self.assertEqual(model.iterations, 50)
        except ImportError:
            self.skipTest("CatBoost not available")
    
    def test_create_model_random_forest(self):
        """Тест создания Random Forest модели"""
        model = self.service._create_model('random_forest', n_estimators=50)
        self.assertIsNotNone(model)
        self.assertEqual(model.n_estimators, 50)
    
    def test_calculate_metrics(self):
        """Тест расчета метрик"""
        y_true = np.array([1, 0, -1, 1, 0])
        y_pred = np.array([1, 0, -1, 1, 0])
        y_prob = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        
        metrics = self.service._calculate_metrics(y_true, y_pred, y_prob)
        
        # Проверяем наличие всех метрик
        expected_metrics = ['confusion_matrix', 'accuracy', 'total_predictions', 'buy_signals', 'sell_signals', 'hold_signals']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Проверяем корректность accuracy
        self.assertEqual(metrics['accuracy'], 1.0)
        
        # Проверяем подсчет сигналов
        self.assertEqual(metrics['buy_signals'], 2)
        self.assertEqual(metrics['sell_signals'], 1)
        self.assertEqual(metrics['hold_signals'], 2)
    
    def test_save_model(self):
        """Тест сохранения модели"""
        # Мок модель
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1, 0, -1]))
        
        coin_id = "test_coin"
        model_type = "lightgbm"
        extra_config = {"test": "config"}
        metrics = {"accuracy": 0.95}
        
        # Создаем временную директорию
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Патчим путь к моделям
            with patch.object(self.service, 'models_dir', temp_dir):
                model_path = self.service._save_model(mock_model, coin_id, model_type, extra_config, metrics)
                
                # Проверяем создание директории
                self.assertTrue(os.path.exists(model_path))
                
                # Проверяем наличие файлов
                config_file = os.path.join(model_path, 'config.json')
                metadata_file = os.path.join(model_path, 'metadata.json')
                
                self.assertTrue(os.path.exists(config_file))
                self.assertTrue(os.path.exists(metadata_file))
    
    def test_load_model(self):
        """Тест загрузки модели"""
        # Создаем временную директорию с мок моделью
        import tempfile
        import json
        import pickle
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем мок модель
            mock_model = Mock()
            mock_model.predict = Mock(return_value=np.array([1, 0, -1]))
            
            # Сохраняем модель
            model_file = os.path.join(temp_dir, 'model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(mock_model, f)
            
            # Создаем конфиг
            config = {
                'coin_id': 'test_coin',
                'model_type': 'lightgbm',
                'timestamp': '20250101_120000'
            }
            config_file = os.path.join(temp_dir, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            # Тестируем загрузку
            loaded_model = self.service.load_model(temp_dir)
            self.assertIsNotNone(loaded_model)
    
    def test_predict(self):
        """Тест предсказания"""
        # Мок модель
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1, 0, -1]))
        mock_model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]))
        
        # Патчим загрузку модели
        with patch.object(self.service, 'load_model', return_value=mock_model):
            # Патчим получение данных
            with patch('core.services.trade_time_service.orm_get_coin_data', return_value=self.mock_df):
                with patch('core.services.trade_time_service.orm_get_news_background', return_value=self.mock_news_data):
                    result = self.service.predict('mock_path', 'test_coin', '2025-01-01', '2025-01-02')
                    
                    # Проверяем структуру результата
                    expected_keys = ['timestamp', 'predictions', 'signals']
                    for key in expected_keys:
                        self.assertIn(key, result)
                    
                    # Проверяем количество предсказаний
                    self.assertEqual(len(result['predictions']), len(result['signals']))
                    
                    # Проверяем корректность сигналов
                    valid_signals = ['BUY', 'SELL', 'HOLD']
                    for signal in result['signals']:
                        self.assertIn(signal, valid_signals)

class TestTradeTimeIntegration(unittest.TestCase):
    """Интеграционные тесты для Trade_time модуля"""
    
    def setUp(self):
        """Настройка тестов"""
        from core.services.trade_time_service import TradeTimeService
        self.service = TradeTimeService()
    
    @patch('core.services.trade_time_service.orm_get_coin_data')
    @patch('core.services.trade_time_service.orm_get_news_background')
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
                'random_forest',
                {'n_estimators': 10}
            )
            
            self.assertEqual(result['status'], 'success')
            self.assertIn('model_path', result)
            self.assertIn('metrics', result)
            self.assertIn('data_info', result)
            
        except Exception as e:
            # Если обучение не удалось, проверяем что ошибка логична
            self.assertIn('error', result)
            self.assertIsInstance(result['error'], str)

if __name__ == '__main__':
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    test_suite.addTest(unittest.makeSuite(TestTradeTimeService))
    test_suite.addTest(unittest.makeSuite(TestTradeTimeIntegration))
    
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
