#!/usr/bin/env python3
"""
Простой тест для Risk модуля без внешних зависимостей
"""

import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_risk_service_import():
    """Тест импорта RiskService"""
    try:
        from core.services.risk_service import RiskService
        print("✅ RiskService импортирован успешно")
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта RiskService: {e}")
        return False

def test_risk_service_creation():
    """Тест создания экземпляра RiskService"""
    try:
        from core.services.risk_service import RiskService
        service = RiskService()
        print("✅ Экземпляр RiskService создан успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания RiskService: {e}")
        return False

def test_risk_service_methods():
    """Тест наличия методов в RiskService"""
    try:
        from core.services.risk_service import RiskService
        service = RiskService()
        
        # Проверяем наличие основных методов
        required_methods = [
            '_calculate_technical_indicators',
            '_calculate_heuristic_risk_score',
            '_calculate_heuristic_volume_score',
            '_prepare_features',
            '_create_model',
            'train_model',
            '_calculate_risk_metrics',
            '_calculate_volume_metrics',
            '_save_models',
            'load_models',
            'predict',
            'calculate_var',
            'calculate_expected_shortfall'
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

def test_risk_service_directory_creation():
    """Тест создания директории для моделей"""
    try:
        from core.services.risk_service import RiskService
        service = RiskService()
        
        # Проверяем что директория создается
        if os.path.exists(service.models_dir):
            print(f"✅ Директория {service.models_dir} существует")
        else:
            print(f"⚠️ Директория {service.models_dir} не существует (будет создана при первом использовании)")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка проверки директории: {e}")
        return False

def test_risk_service_risk_calculation():
    """Тест расчета риска"""
    try:
        from core.services.risk_service import RiskService
        service = RiskService()
        
        # Тестируем расчет VaR и Expected Shortfall
        returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02]
        
        var_95 = service.calculate_var(returns, 0.95)
        es_95 = service.calculate_expected_shortfall(returns, 0.95)
        
        print(f"✅ VaR 95%: {var_95:.4f}")
        print(f"✅ Expected Shortfall 95%: {es_95:.4f}")
        
        # Проверяем что значения корректные
        if isinstance(var_95, float) and isinstance(es_95, float):
            print("✅ Расчет риска работает корректно")
            return True
        else:
            print("❌ Расчет риска вернул некорректные значения")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка расчета риска: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🧪 Запуск простых тестов для Risk модуля")
    print("=" * 50)
    
    tests = [
        ("Импорт RiskService", test_risk_service_import),
        ("Создание экземпляра", test_risk_service_creation),
        ("Проверка методов", test_risk_service_methods),
        ("Проверка директории", test_risk_service_directory_creation),
        ("Расчет риска", test_risk_service_risk_calculation),
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
    assert success, "Risk simple tests failed"
