"""
Тесты для новых таблиц БД: artifacts, backtests, pipelines
"""
import sys
import os
from datetime import datetime, timedelta
import json

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_artifacts_table_structure():
    """Тест структуры таблицы artifacts"""
    try:
        from core.database.models.main_models import Artifact
        from core.database.models.ML_models import Agent
        
        # Проверяем что класс существует
        assert Artifact is not None
        assert hasattr(Artifact, '__tablename__')
        assert Artifact.__tablename__ == 'artifacts'
        
        # Проверяем основные поля
        assert hasattr(Artifact, 'id')
        assert hasattr(Artifact, 'agent_id')
        assert hasattr(Artifact, 'version')
        assert hasattr(Artifact, 'path')
        assert hasattr(Artifact, 'type')
        assert hasattr(Artifact, 'size_bytes')
        assert hasattr(Artifact, 'checksum')
        assert hasattr(Artifact, 'created_at')
        
        # Проверяем связь с Agent
        assert hasattr(Artifact, 'agent')
        
        print("✅ Artifacts table structure test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️ Import error in artifacts test: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in artifacts test: {e}")
        return False


def test_backtests_table_structure():
    """Тест структуры таблицы backtests"""
    try:
        from core.database.models.main_models import Backtest
        
        # Проверяем что класс существует
        assert Backtest is not None
        assert hasattr(Backtest, '__tablename__')
        assert Backtest.__tablename__ == 'backtests'
        
        # Проверяем основные поля
        assert hasattr(Backtest, 'id')
        assert hasattr(Backtest, 'pipeline_id')
        assert hasattr(Backtest, 'name')
        assert hasattr(Backtest, 'config_json')
        assert hasattr(Backtest, 'timeframe')
        assert hasattr(Backtest, 'start_date')
        assert hasattr(Backtest, 'end_date')
        assert hasattr(Backtest, 'coins')
        assert hasattr(Backtest, 'metrics_json')
        assert hasattr(Backtest, 'artifacts')
        assert hasattr(Backtest, 'status')
        assert hasattr(Backtest, 'progress')
        assert hasattr(Backtest, 'error_message')
        assert hasattr(Backtest, 'created_at')
        assert hasattr(Backtest, 'completed_at')
        
        # Проверяем связь с Pipeline
        assert hasattr(Backtest, 'pipeline')
        
        print("✅ Backtests table structure test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️ Import error in backtests test: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in backtests test: {e}")
        return False


def test_pipelines_table_structure():
    """Тест структуры таблицы pipelines"""
    try:
        from core.database.models.main_models import Pipeline
        
        # Проверяем что класс существует
        assert Pipeline is not None
        assert hasattr(Pipeline, '__tablename__')
        assert Pipeline.__tablename__ == 'pipelines'
        
        # Проверяем основные поля
        assert hasattr(Pipeline, 'id')
        assert hasattr(Pipeline, 'name')
        assert hasattr(Pipeline, 'description')
        assert hasattr(Pipeline, 'config_json')
        assert hasattr(Pipeline, 'is_template')
        assert hasattr(Pipeline, 'created_by')
        assert hasattr(Pipeline, 'created_at')
        assert hasattr(Pipeline, 'updated_at')
        
        # Проверяем связи
        assert hasattr(Pipeline, 'user')
        assert hasattr(Pipeline, 'backtests')
        
        print("✅ Pipelines table structure test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️ Import error in pipelines test: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in pipelines test: {e}")
        return False


def test_news_background_table_structure():
    """Тест структуры таблицы news_background"""
    try:
        from core.database.models.main_models import NewsBackground
        
        # Проверяем что класс существует
        assert NewsBackground is not None
        assert hasattr(NewsBackground, '__tablename__')
        assert NewsBackground.__tablename__ == 'news_background'
        
        # Проверяем основные поля
        assert hasattr(NewsBackground, 'id')
        assert hasattr(NewsBackground, 'coin_id')
        assert hasattr(NewsBackground, 'timestamp')
        assert hasattr(NewsBackground, 'score')
        assert hasattr(NewsBackground, 'source_count')
        assert hasattr(NewsBackground, 'sources_breakdown')
        assert hasattr(NewsBackground, 'window_hours')
        assert hasattr(NewsBackground, 'decay_factor')
        assert hasattr(NewsBackground, 'created_at')
        
        # Проверяем связь с Coin
        assert hasattr(NewsBackground, 'coin')
        
        print("✅ NewsBackground table structure test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️ Import error in news_background test: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in news_background test: {e}")
        return False


def test_agent_artifacts_relationship():
    """Тест связи Agent с Artifacts"""
    try:
        from core.database.models.ML_models import Agent
        from core.database.models.main_models import Artifact
        
        # Проверяем что связь существует
        assert hasattr(Agent, 'artifacts')
        
        print("✅ Agent-Artifacts relationship test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️ Import error in agent-artifacts relationship test: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in agent-artifacts relationship test: {e}")
        return False


def test_metrics_module_functions():
    """Тест функций модуля метрик"""
    try:
        from core.utils.metrics import (
            calculate_regression_metrics,
            calculate_classification_metrics,
            calculate_risk_metrics,
            calculate_trading_metrics,
            calculate_portfolio_metrics
        )
        
        # Проверяем что функции существуют
        assert calculate_regression_metrics is not None
        assert calculate_classification_metrics is not None
        assert calculate_risk_metrics is not None
        assert calculate_trading_metrics is not None
        assert calculate_portfolio_metrics is not None
        
        # Тестируем calculate_regression_metrics
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.1, 3.9]
        reg_metrics = calculate_regression_metrics(y_true, y_pred)
        
        assert 'mae' in reg_metrics
        assert 'rmse' in reg_metrics
        assert 'mape' in reg_metrics
        assert 'direction_hit_rate' in reg_metrics
        
        # Тестируем calculate_classification_metrics
        y_true_bin = [0, 1, 0, 1, 0]
        y_pred_bin = [0, 1, 0, 0, 1]
        y_score_bin = [0.1, 0.9, 0.2, 0.4, 0.6]
        class_metrics = calculate_classification_metrics(y_true_bin, y_pred_bin, y_score_bin)
        
        assert 'confusion_matrix' in class_metrics
        assert 'precision_recall_f1' in class_metrics
        assert 'labels' in class_metrics
        assert 'roc_auc' in class_metrics
        assert 'pr_auc' in class_metrics
        
        # Тестируем calculate_risk_metrics
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        risk_metrics = calculate_risk_metrics(returns)
        
        assert 'var' in risk_metrics
        assert 'expected_shortfall' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'volatility' in risk_metrics
        
        # Тестируем calculate_trading_metrics
        trading_metrics = calculate_trading_metrics(returns)
        
        assert 'total_return' in trading_metrics
        assert 'sharpe_ratio' in trading_metrics
        assert 'sortino_ratio' in trading_metrics
        assert 'win_rate' in trading_metrics
        assert 'max_drawdown' in trading_metrics
        assert 'volatility' in trading_metrics
        
        # Тестируем calculate_portfolio_metrics
        returns_per_asset = [[0.01, 0.02, -0.01], [0.005, -0.01, 0.015]]
        portfolio_metrics = calculate_portfolio_metrics(returns_per_asset)
        
        assert 'portfolio' in portfolio_metrics
        assert 'per_asset' in portfolio_metrics
        assert 'equity_curve' in portfolio_metrics
        
        print("✅ Metrics module functions test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️ Import error in metrics test: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in metrics test: {e}")
        return False


def main():
    """Запуск всех тестов"""
    print("🧪 Running database models and metrics tests...")
    
    tests = [
        test_artifacts_table_structure,
        test_backtests_table_structure,
        test_pipelines_table_structure,
        test_news_background_table_structure,
        test_agent_artifacts_relationship,
        test_metrics_module_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed")
        return 1


if __name__ == '__main__':
    success = main()
    assert success, "Database models tests failed"
