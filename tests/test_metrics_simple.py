#!/usr/bin/env python3
"""
Упрощенные тесты для Metrics модуля без PyTorch зависимостей
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestMetricsSimple(unittest.TestCase):
    """Упрощенные тесты для Metrics модуля"""
    
    def setUp(self):
        """Настройка тестов"""
        try:
            from core.utils.metrics import (
                mean_absolute_error, root_mean_squared_error, 
                mean_absolute_percentage_error, direction_hit_rate,
                confusion_matrix, precision_recall_f1,
                roc_auc_score, pr_auc_score, value_at_risk, 
                expected_shortfall, win_rate, turnover_rate,
                exposure_stats, equity_curve, max_drawdown,
                sharpe_ratio, sortino_ratio, aggregate_returns_equal_weight,
                calculate_portfolio_metrics, calculate_regression_metrics,
                calculate_classification_metrics, calculate_risk_metrics,
                calculate_trading_metrics
            )
            self.metrics_module = sys.modules['core.utils.metrics']
        except ImportError as e:
            self.skipTest(f"Metrics модуль недоступен: {e}")
    
    def test_metrics_functions_exist(self):
        """Тест наличия основных функций metrics"""
        required_functions = [
            'mean_absolute_error',
            'root_mean_squared_error', 
            'mean_absolute_percentage_error',
            'direction_hit_rate',
            'confusion_matrix',
            'precision_recall_f1',
            'roc_auc_score',
            'pr_auc_score',
            'value_at_risk',
            'expected_shortfall',
            'win_rate',
            'turnover_rate',
            'exposure_stats',
            'equity_curve',
            'max_drawdown',
            'sharpe_ratio',
            'sortino_ratio',
            'aggregate_returns_equal_weight',
            'calculate_portfolio_metrics',
            'calculate_regression_metrics',
            'calculate_classification_metrics',
            'calculate_risk_metrics',
            'calculate_trading_metrics'
        ]
        
        for func_name in required_functions:
            self.assertTrue(hasattr(self.metrics_module, func_name), 
                          f"Функция {func_name} не найдена")
    
    def test_basic_regression_metrics(self):
        """Тест базовых метрик регрессии"""
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.1, 3.9]
        
        # Тестируем MAE
        mae = self.metrics_module.mean_absolute_error(y_true, y_pred)
        self.assertIsInstance(mae, (int, float))
        self.assertGreaterEqual(mae, 0)
        
        # Тестируем RMSE
        rmse = self.metrics_module.root_mean_squared_error(y_true, y_pred)
        self.assertIsInstance(rmse, (int, float))
        self.assertGreaterEqual(rmse, 0)
        
        # Тестируем MAPE
        mape = self.metrics_module.mean_absolute_percentage_error(y_true, y_pred)
        self.assertIsInstance(mape, (int, float))
        self.assertGreaterEqual(mape, 0)
    
    def test_basic_classification_metrics(self):
        """Тест базовых метрик классификации"""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1]
        
        # Тестируем confusion matrix
        cm = self.metrics_module.confusion_matrix(y_true, y_pred, [0, 1])
        self.assertIsInstance(cm, list)
        self.assertEqual(len(cm), 2)
        
        # Тестируем precision_recall_f1
        prf = self.metrics_module.precision_recall_f1(cm)
        self.assertIsInstance(prf, dict)
        self.assertIn('macro', prf)
        self.assertIn('micro', prf)
    
    def test_basic_risk_metrics(self):
        """Тест базовых метрик риска"""
        returns = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        
        # Тестируем VaR
        var_95 = self.metrics_module.value_at_risk(returns, 0.95)
        self.assertIsInstance(var_95, (int, float))
        
        # Тестируем Expected Shortfall
        es = self.metrics_module.expected_shortfall(returns, 0.95)
        self.assertIsInstance(es, (int, float))
    
    def test_basic_trading_metrics(self):
        """Тест базовых торговых метрик"""
        returns = [-0.01, 0.02, -0.005, 0.015, -0.02, 0.01]
        positions = [0.1, 0.2, 0.15, 0.25, 0.3]
        
        # Тестируем win rate
        wr = self.metrics_module.win_rate(returns)
        self.assertIsInstance(wr, (int, float))
        self.assertGreaterEqual(wr, 0)
        self.assertLessEqual(wr, 1)
        
        # Тестируем turnover rate
        turnover = self.metrics_module.turnover_rate(positions)
        self.assertIsInstance(turnover, (int, float))
        self.assertGreaterEqual(turnover, 0)
    
    def test_basic_portfolio_metrics(self):
        """Тест базовых метрик портфеля"""
        returns = [0.01, -0.005, 0.02, -0.01]
        
        # Тестируем equity curve
        equity = self.metrics_module.equity_curve(returns, 1000.0)
        self.assertIsInstance(equity, list)
        self.assertEqual(len(equity), len(returns) + 1)
        
        # Тестируем max drawdown
        mdd = self.metrics_module.max_drawdown(equity)
        self.assertIsInstance(mdd, (int, float))
        self.assertLessEqual(mdd, 0)  # Drawdown should be negative
    
    def test_utility_functions(self):
        """Тест утилитарных функций"""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 1.9, 3.1, 3.9, 5.1]
        
        # Тестируем calculate_regression_metrics
        reg_metrics = self.metrics_module.calculate_regression_metrics(y_true, y_pred)
        self.assertIsInstance(reg_metrics, dict)
        self.assertIn('mae', reg_metrics)
        self.assertIn('rmse', reg_metrics)
        
        # Тестируем calculate_risk_metrics
        returns = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        positions = [0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4]
        risk_metrics = self.metrics_module.calculate_risk_metrics(returns, positions, 0.95)
        self.assertIsInstance(risk_metrics, dict)
        self.assertIn('var', risk_metrics)
        self.assertIn('expected_shortfall', risk_metrics)

def main():
    """Запуск тестов"""
    print("🧪 Запуск упрощенных тестов Metrics модуля...")
    
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    loader = unittest.TestLoader()
    test_suite.addTest(loader.loadTestsFromTestCase(TestMetricsSimple))
    
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
    assert exit_code == 0, "Metrics simple tests failed"
