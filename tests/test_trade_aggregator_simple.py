#!/usr/bin/env python3
"""
Простой структурный тест для Trade Aggregator модуля
"""

import sys
import os

def test_import():
    """Тест импорта TradeAggregatorService"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.services.trade_aggregator_service import TradeAggregatorService
        print("✅ TradeAggregatorService успешно импортирован")
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта TradeAggregatorService: {e}")
        return False

def test_class_creation():
    """Тест создания экземпляра класса"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.services.trade_aggregator_service import TradeAggregatorService
        
        service = TradeAggregatorService()
        print("✅ Экземпляр TradeAggregatorService успешно создан")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания экземпляра: {e}")
        return False

def test_methods_exist():
    """Тест наличия необходимых методов"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.services.trade_aggregator_service import TradeAggregatorService
        
        service = TradeAggregatorService()
        
        required_methods = [
            'train_model',
            'evaluate_model', 
            'predict',
            '_calculate_portfolio_metrics',
            '_aggregate_signals',
            '_apply_risk_management',
            '_create_ml_model',
            '_calculate_max_drawdown',
            '_save_model',
            'load_model'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(service, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Отсутствуют методы: {missing_methods}")
            return False
        else:
            print("✅ Все необходимые методы присутствуют")
            return True
            
    except Exception as e:
        print(f"❌ Ошибка проверки методов: {e}")
        return False

def test_config_structure():
    """Тест структуры конфигурации по умолчанию"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.services.trade_aggregator_service import TradeAggregatorService
        
        service = TradeAggregatorService()
        
        required_config_keys = [
            'mode',
            'weights',
            'thresholds', 
            'risk_limits',
            'portfolio'
        ]
        
        missing_keys = []
        for key in required_config_keys:
            if key not in service.default_config:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ Отсутствуют ключи конфигурации: {missing_keys}")
            return False
        else:
            print("✅ Все ключи конфигурации присутствуют")
            return True
            
    except Exception as e:
        print(f"❌ Ошибка проверки конфигурации: {e}")
        return False

def main():
    """Основная функция"""
    print("🧪 Простой структурный тест Trade Aggregator модуля...")
    
    tests = [
        test_import,
        test_class_creation,
        test_methods_exist,
        test_config_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Результат: {passed}/{total} тестов прошли успешно")
    
    if passed == total:
        print("✅ Все структурные тесты Trade Aggregator модуля прошли успешно!")
        return 0
    else:
        print("❌ Некоторые структурные тесты Trade Aggregator модуля провалились")
        return 1

if __name__ == '__main__':
    success = main()
    assert success, "Trade aggregator simple tests failed"
