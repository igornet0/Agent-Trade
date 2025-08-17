#!/usr/bin/env python3
"""
Базовый тест для проверки структуры модулей без внешних зависимостей
"""

import sys
import os
import ast

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_file_syntax(file_path):
    """Тест синтаксиса Python файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Парсим AST для проверки синтаксиса
        ast.parse(content)
        return True
    except SyntaxError as e:
        print(f"❌ Синтаксическая ошибка в {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка чтения {file_path}: {e}")
        return False

def test_file_structure(file_path, required_methods):
    """Тест структуры файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем наличие методов
        for method in required_methods:
            if method in content:
                print(f"✅ Метод {method} найден в {os.path.basename(file_path)}")
            else:
                print(f"❌ Метод {method} не найден в {os.path.basename(file_path)}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Ошибка проверки структуры {file_path}: {e}")
        return False

def test_trade_time_service_structure():
    """Тест структуры TradeTimeService"""
    file_path = "src/core/services/trade_time_service.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем синтаксис
    if not test_file_syntax(file_path):
        return False
    
    # Проверяем структуру
    required_methods = [
        'class TradeTimeService',
        'def _calculate_technical_indicators',
        'def _prepare_features',
        'def _create_model',
        'def train_model',
        'def _calculate_metrics',
        'def _save_model',
        'def load_model',
        'def predict'
    ]
    
    return test_file_structure(file_path, required_methods)

def test_risk_service_structure():
    """Тест структуры RiskService"""
    file_path = "src/core/services/risk_service.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем синтаксис
    if not test_file_syntax(file_path):
        return False
    
    # Проверяем структуру
    required_methods = [
        'class RiskService',
        'def _calculate_technical_indicators',
        'def _calculate_heuristic_risk_score',
        'def _calculate_heuristic_volume_score',
        'def _prepare_features',
        'def _create_model',
        'def train_model',
        'def _calculate_risk_metrics',
        'def _calculate_volume_metrics',
        'def _save_models',
        'def load_models',
        'def predict',
        'def calculate_var',
        'def calculate_expected_shortfall'
    ]
    
    return test_file_structure(file_path, required_methods)

def test_celery_tasks_structure():
    """Тест структуры Celery задач"""
    file_path = "src/backend/celery_app/tasks.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем синтаксис
    if not test_file_syntax(file_path):
        return False
    
    # Проверяем структуру
    required_methods = [
        'def train_trade_time_task',
        'def evaluate_trade_time_task',
        'def train_risk_task',
        'def evaluate_risk_task'
    ]
    
    return test_file_structure(file_path, required_methods)

def test_api_endpoints_structure():
    """Тест структуры API endpoints"""
    file_path = "src/backend/app/routers/apidb_agent/router.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем синтаксис
    if not test_file_syntax(file_path):
        return False
    
    # Проверяем структуру
    required_methods = [
        'def train_trade_time',
        'def evaluate_trade_time',
        'def get_trade_time_models',
        'def predict_trade_time',
        'def train_risk',
        'def evaluate_risk',
        'def get_risk_models',
        'def predict_risk'
    ]
    
    return test_file_structure(file_path, required_methods)

def test_schemas_structure():
    """Тест структуры схем"""
    file_path = "src/backend/app/schemas/agent.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем синтаксис
    if not test_file_syntax(file_path):
        return False
    
    # Проверяем структуру
    required_classes = [
        'class TradeTimeTrainConfig',
        'class RiskTrainConfig',
        'class TradeTimeModel',
        'class RiskModel',
        'class TradeTimePrediction',
        'class RiskPrediction'
    ]
    
    return test_file_structure(file_path, required_classes)

def test_frontend_components():
    """Тест структуры фронтенд компонентов"""
    components = [
        ("frontend/src/components/profile/TradeTimeTrainPanel.jsx", [
            'const TradeTimeTrainPanel',
            'useState',
            'handleChange',
            'model_type',
            'n_estimators'
        ]),
        ("frontend/src/components/profile/RiskTrainPanel.jsx", [
            'const RiskTrainPanel',
            'useState',
            'handleChange',
            'risk_weight',
            'volume_weight'
        ])
    ]
    
    results = []
    for file_path, required_elements in components:
        if not os.path.exists(file_path):
            print(f"❌ Файл {file_path} не найден")
            results.append(False)
            continue
        
        # Проверяем структуру JSX файла
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for element in required_elements:
                if element in content:
                    print(f"✅ Элемент {element} найден в {os.path.basename(file_path)}")
                else:
                    print(f"❌ Элемент {element} не найден в {os.path.basename(file_path)}")
                    results.append(False)
                    break
            else:
                results.append(True)
                
        except Exception as e:
            print(f"❌ Ошибка проверки {file_path}: {e}")
            results.append(False)
    
    return all(results)

def main():
    """Основная функция тестирования"""
    print("🧪 Запуск базовых тестов для модулей")
    print("=" * 50)
    
    tests = [
        ("Структура TradeTimeService", test_trade_time_service_structure),
        ("Структура RiskService", test_risk_service_structure),
        ("Структура Celery задач", test_celery_tasks_structure),
        ("Структура API endpoints", test_api_endpoints_structure),
        ("Структура схем", test_schemas_structure),
        ("Фронтенд компоненты", test_frontend_components),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Тест: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} прошел успешно")
            else:
                print(f"❌ {test_name} провален")
        except Exception as e:
            print(f"❌ {test_name} завершился с ошибкой: {e}")
    
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
    sys.exit(main())
