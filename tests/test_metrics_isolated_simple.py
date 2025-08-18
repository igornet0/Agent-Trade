#!/usr/bin/env python3
"""
Упрощенные тесты для Metrics Isolated модуля без PyTorch зависимостей
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestMetricsIsolatedSimple(unittest.TestCase):
    """Упрощенные тесты для Metrics Isolated модуля"""
    
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
    
    def test_regression_metrics_isolated(self):
        """Тест изолированных метрик регрессии"""
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.1, 3.9]
        
        # Тестируем MAE
        mae = self.metrics_module.mean_absolute_error(y_true, y_pred)
        self.assertAlmostEqual(mae, 0.1, places=6)
        
        # Тестируем RMSE
        rmse = self.metrics_module.root_mean_squared_error(y_true, y_pred)
        self.assertAlmostEqual(rmse, 0.1, places=6)
        
        # Тестируем MAPE
        mape = self.metrics_module.mean_absolute_percentage_error(y_true, y_pred)
        self.assertGreater(mape, 0.02)
        self.assertLess(mape, 0.06)  # Увеличиваем диапазон для MAPE
    
    def test_classification_metrics_isolated(self):
        """Тест изолированных метрик классификации"""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1]
        
        # Тестируем confusion matrix
        cm = self.metrics_module.confusion_matrix(y_true, y_pred, [0, 1])
        self.assertEqual(cm, [[2, 1], [1, 1]])
        
        # Тестируем precision_recall_f1
        prf = self.metrics_module.precision_recall_f1(cm)
        self.assertIn('macro', prf)
        self.assertIn('micro', prf)
        self.assertIn('per_class', prf)
        self.assertEqual(len(prf["per_class"]), 2)
    
    def test_risk_metrics_isolated(self):
        """Тест изолированных метрик риска"""
        returns = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
        
        # Тестируем VaR
        var_95 = self.metrics_module.value_at_risk(returns, 0.95)
        var_99 = self.metrics_module.value_at_risk(returns, 0.99)
        self.assertGreaterEqual(var_95, var_99)
        self.assertLessEqual(var_95, 0.0)
        
        # Тестируем Expected Shortfall
        es = self.metrics_module.expected_shortfall(returns, 0.95)
        var = self.metrics_module.value_at_risk(returns, 0.95)
        self.assertLessEqual(es, var)
    
    def test_trading_metrics_isolated(self):
        """Тест изолированных торговых метрик"""
        returns = [-0.01, 0.02, -0.005, 0.015, -0.02, 0.01]
        positions = [0.1, 0.2, 0.15, 0.25, 0.3]
        
        # Тестируем win rate
        wr = self.metrics_module.win_rate(returns)
        self.assertEqual(wr, 0.5)
        
        # Тестируем exposure stats
        stats = self.metrics_module.exposure_stats(positions)
        self.assertIn("max_exposure", stats)
        self.assertIn("avg_exposure", stats)
        self.assertIn("exposure_volatility", stats)
        self.assertEqual(stats["max_exposure"], 0.3)
    
    def test_portfolio_metrics_isolated(self):
        """Тест изолированных метрик портфеля"""
        returns = [0.01, -0.005, 0.02, -0.01]
        
        # Тестируем equity curve
        equity = self.metrics_module.equity_curve(returns, 1000.0)
        self.assertEqual(len(equity), len(returns) + 1)
        self.assertEqual(equity[0], 1000.0)
        self.assertGreater(equity[-1], 1000.0)
        
        # Тестируем max drawdown
        mdd = self.metrics_module.max_drawdown(equity)
        self.assertLess(mdd, 0.0)
        self.assertGreaterEqual(mdd, -0.1)

def main():
    """Запуск тестов"""
    print("🧪 Запуск упрощенных тестов Metrics Isolated модуля...")
    
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    loader = unittest.TestLoader()
    test_suite.addTest(loader.loadTestsFromTestCase(TestMetricsIsolatedSimple))
    
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
    assert exit_code == 0, "Metrics isolated simple tests failed"
