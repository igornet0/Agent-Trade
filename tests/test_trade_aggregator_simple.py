#!/usr/bin/env python3
"""
Упрощенные тесты для Trade Aggregator модуля без PyTorch зависимостей
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestTradeAggregatorServiceSimple(unittest.TestCase):
    """Упрощенные тесты для TradeAggregatorService без PyTorch"""
    
    def setUp(self):
        """Настройка тестов"""
        # Патчим импорт PyTorch чтобы избежать конфликтов
        with patch.dict('sys.modules', {'torch': Mock()}):
            try:
                from core.services.trade_aggregator_service import TradeAggregatorService
                self.service = TradeAggregatorService()
            except ImportError as e:
                self.skipTest(f"TradeAggregatorService недоступен: {e}")
        
        # Мок данные
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
        pred_time_signals = [1, 1, 0, -1, 1]
        trade_time_signals = [0, 1, 1, 0, -1]
        risk_signals = [0.8, 0.3, 0.9, 0.2, 0.7]
        config = {'signal_weights': {'pred_time': 0.4, 'trade_time': 0.4, 'risk': 0.2}}
        
        # Тестируем агрегацию
        result = self.service._aggregate_signals(pred_time_signals, trade_time_signals, risk_signals, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('decision', result)
        self.assertIn('confidence', result)
        self.assertIn('aggregated_score', result)
    
    def test_calculate_portfolio_metrics_basic(self):
        """Базовый тест расчета метрик портфеля"""
        positions = [
            {'coin_id': 'BTC', 'quantity': 1.0, 'avg_price': 50000, 'unrealized_pnl': 5000, 'size': 1.0, 'entry_price': 50000},
            {'coin_id': 'ETH', 'quantity': 10.0, 'avg_price': 3000, 'unrealized_pnl': 2000, 'size': 10.0, 'entry_price': 3000}
        ]
        
        balance = 100000
        
        # Тестируем расчет метрик
        result = self.service._calculate_portfolio_metrics(positions, balance)
        
        self.assertIsInstance(result, dict)
        self.assertIn('total_value', result)
        self.assertIn('total_pnl', result)
        self.assertIn('total_return', result)
    
    def test_apply_risk_management_basic(self):
        """Базовый тест применения риск-менеджмента"""
        decision = 1  # Buy signal
        position_size = 0.1  # 10% of portfolio
        portfolio_metrics = {'total_value': 100000, 'total_return': 0.05, 'exposure': 80000}
        config = {'max_position_size': 0.2, 'stop_loss_pct': 0.05}
        
        # Тестируем применение риск-менеджмента
        result = self.service._apply_risk_management(decision, position_size, portfolio_metrics, config)
        
        self.assertIsInstance(result, dict)
        self.assertIn('adjusted_signal', result)
        self.assertIn('position_size', result)
        self.assertIn('stop_loss', result)
    
    def test_calculate_max_drawdown_basic(self):
        """Базовый тест расчета максимальной просадки"""
        coin_data = [
            {'close': 100}, {'close': 98}, {'close': 102}, {'close': 99}, 
            {'close': 105}, {'close': 101}, {'close': 103}, {'close': 100}
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

def main():
    """Запуск тестов"""
    print("🧪 Запуск упрощенных тестов Trade Aggregator модуля...")
    
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    loader = unittest.TestLoader()
    test_suite.addTest(loader.loadTestsFromTestCase(TestTradeAggregatorServiceSimple))
    
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
    assert exit_code == 0, "Trade aggregator simple tests failed"
