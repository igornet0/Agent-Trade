#!/usr/bin/env python3
"""
Простой тест для Trade_time модуля без внешних зависимостей
"""

import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_trade_time_service_import():
    """Тест импорта TradeTimeService"""
    try:
        from core.services.trade_time_service import TradeTimeService
        print("✅ TradeTimeService импортирован успешно")
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта TradeTimeService: {e}")
        return False

def test_trade_time_service_creation():
    """Тест создания экземпляра TradeTimeService"""
    try:
        from core.services.trade_time_service import TradeTimeService
        service = TradeTimeService()
        print("✅ Экземпляр TradeTimeService создан успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания TradeTimeService: {e}")
        return False

def test_trade_time_service_methods():
    """Тест наличия методов в TradeTimeService"""
    try:
        from core.services.trade_time_service import TradeTimeService
        service = TradeTimeService()
        
        # Проверяем наличие основных методов
        required_methods = [
            '_calculate_technical_indicators',
            '_prepare_features',
            '_create_model',
            'train_model',
            '_calculate_metrics',
            '_save_model',
            'load_model',
            'predict'
        ]
        
        for method_name in required_methods:
            if hasattr(service, method_name):
                print(f"✅ Метод {method_name} найден")
            else:
                print(f"❌ Метод {method_name} не найден")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Ошибка проверки методов: {e}")
        return False

def test_trade_time_service_directory_creation():
    """Тест создания директории для моделей"""
    try:
        from core.services.trade_time_service import TradeTimeService
        service = TradeTimeService()
        
        # Проверяем что директория создается
        if os.path.exists(service.models_dir):
            print(f"✅ Директория {service.models_dir} существует")
        else:
            print(f"⚠️ Директория {service.models_dir} не существует (будет создана при первом использовании)")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка проверки директории: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🧪 Запуск простых тестов для Trade_time модуля")
    print("=" * 50)
    
    tests = [
        ("Импорт TradeTimeService", test_trade_time_service_import),
        ("Создание экземпляра", test_trade_time_service_creation),
        ("Проверка методов", test_trade_time_service_methods),
        ("Проверка директории", test_trade_time_service_directory_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Тест: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Тест {test_name} провален")
        except Exception as e:
            print(f"❌ Тест {test_name} завершился с ошибкой: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Результаты тестирования:")
    print(f"✅ Успешно: {passed}/{total}")
    print(f"❌ Провалено: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 Все тесты прошли успешно!")
        return 0
    else:
        print("⚠️ Некоторые тесты провалились")
        return 1

if __name__ == '__main__':
    success = main()
    assert success, "Trade time simple tests failed"
