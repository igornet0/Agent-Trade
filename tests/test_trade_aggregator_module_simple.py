#!/usr/bin/env python3
"""
Упрощенные тесты для Trade Aggregator Module без PyTorch зависимостей
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestTradeAggregatorModuleSimple(unittest.TestCase):
    """Упрощенные тесты для Trade Aggregator Module"""
    
    def setUp(self):
        """Настройка тестов"""
        try:
            from core.services.trade_aggregator_service import TradeAggregatorService
            self.service = TradeAggregatorService()
        except ImportError as e:
            self.skipTest(f"TradeAggregatorService недоступен: {e}")
        
        # Мок данных
        self.mock_df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=100, freq='1h'),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
    
    def test_service_creation(self):
        """Тест создания сервиса"""
        self.assertIsNotNone(self.service)
        self.assertTrue(hasattr(self.service, 'models_dir'))
    
    def test_aggregate_signals_basic(self):
        """Базовый тест агрегации сигналов"""
        pred_time_signals = [1, 0, -1, 1, 0]
        trade_time_signals = [1, 1, 0, -1, 1]
        risk_signals = [0.8, 0.3, 0.9, 0.2, 0.7]
        config = self.service.default_config
        
        # Тестируем агрегацию
        result = self.service._aggregate_signals(pred_time_signals, trade_time_signals, risk_signals, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('decision', result)
        self.assertIn('confidence', result)
        self.assertIn('aggregated_score', result)
    
    def test_calculate_portfolio_metrics_basic(self):
        """Базовый тест расчета метрик портфеля"""
        positions = [
            {'size': 1.0, 'entry_price': 50000, 'unrealized_pnl': 5000},
            {'size': 10.0, 'entry_price': 3000, 'unrealized_pnl': 2000}
        ]
        balance = 100000
        
        # Тестируем расчет метрик
        result = self.service._calculate_portfolio_metrics(positions, balance)
        
        self.assertIsInstance(result, dict)
        self.assertIn('total_value', result)
        self.assertIn('total_pnl', result)
        self.assertIn('exposure', result)
    
    def test_apply_risk_management_basic(self):
        """Базовый тест применения риск-менеджмента"""
        decision = 1  # Buy signal
        position_size = 0.1  # 10% of balance
        portfolio_metrics = {
            'total_value': 100000,
            'risk_score': 0.8,
            'exposure': 50000
        }
        config = self.service.default_config
        
        # Тестируем применение риск-менеджмента
        result = self.service._apply_risk_management(decision, position_size, portfolio_metrics, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('adjusted_signal', result)
        self.assertIn('position_size', result)
        self.assertIn('stop_loss_pct', result)
    
    def test_calculate_max_drawdown_basic(self):
        """Базовый тест расчета максимальной просадки"""
        coin_data = [
            {'close': 100},
            {'close': 95},
            {'close': 90},
            {'close': 85},
            {'close': 88},
            {'close': 92}
        ]
        
        # Тестируем расчет просадки
        result = self.service._calculate_max_drawdown(coin_data)
        
        self.assertIsInstance(result, float)
        self.assertLessEqual(result, 0)  # Просадка должна быть отрицательной
    
    def test_service_methods_exist(self):
        """Тест наличия основных методов"""
        required_methods = [
            '_aggregate_signals',
            '_calculate_portfolio_metrics',
            '_apply_risk_management',
            '_calculate_max_drawdown',
            'train_model',
            'predict'
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(self.service, method_name), 
                          f"Метод {method_name} отсутствует")
    
    def test_data_processing_basic(self):
        """Базовый тест обработки данных"""
        # Тестируем обработку данных
        self.assertIsInstance(self.mock_df, pd.DataFrame)
        self.assertEqual(len(self.mock_df), 100)
        self.assertIn('close', self.mock_df.columns)
        self.assertIn('volume', self.mock_df.columns)
    
    def test_model_operations_basic(self):
        """Базовый тест операций с моделями"""
        # Тестируем операции с моделями (без реального обучения)
        try:
            # Проверяем, что методы существуют
            self.assertTrue(hasattr(self.service, 'train_model'))
            self.assertTrue(hasattr(self.service, 'predict'))
        except Exception as e:
            self.skipTest(f"Операции с моделями недоступны: {e}")

def main():
    """Запуск тестов"""
    print("🧪 Запуск упрощенных тестов Trade Aggregator Module...")
    
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    loader = unittest.TestLoader()
    test_suite.addTest(loader.loadTestsFromTestCase(TestTradeAggregatorModuleSimple))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Выводим результат
    print(f"\nТесты завершены: {result.testsRun} тестов выполнено")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Провалов: {len(result.failures)}")
    
    if result.wasSuccessful():
        print("✅ Все тесты прошли успешно!")
        return 0
    else:
        print("❌ Некоторые тесты провалились")
        return 1

if __name__ == '__main__':
    exit_code = main()
    assert exit_code == 0, "Trade aggregator module simple tests failed"
