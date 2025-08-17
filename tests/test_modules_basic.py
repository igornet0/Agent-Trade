#!/usr/bin/env python3
"""
Базовые тесты для проверки структуры всех модулей
"""

import sys
import os
import ast

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_file_syntax(file_path):
    """Проверка синтаксиса Python файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True
    except SyntaxError as e:
        print(f"❌ Синтаксическая ошибка в {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка чтения {file_path}: {e}")
        return False

def test_file_structure(file_path, required_elements):
    """Проверка наличия необходимых элементов в файле"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ В {file_path} отсутствуют: {missing_elements}")
            return False
        else:
            print(f"✅ {file_path} содержит все необходимые элементы")
            return True
    except Exception as e:
        print(f"❌ Ошибка проверки {file_path}: {e}")
        return False

def test_trade_aggregator_service():
    """Тест Trade Aggregator сервиса"""
    file_path = "src/core/services/trade_aggregator_service.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем синтаксис
    if not test_file_syntax(file_path):
        return False
    
    # Проверяем структуру
    required_elements = [
        "class TradeAggregatorService",
        "def train_model",
        "def evaluate_model",
        "def predict",
        "_calculate_portfolio_metrics",
        "_aggregate_signals",
        "_apply_risk_management",
        "_create_ml_model",
        "_calculate_max_drawdown",
        "_save_model",
        "load_model"
    ]
    
    return test_file_structure(file_path, required_elements)

def test_celery_tasks():
    """Тест Celery задач"""
    file_path = "src/backend/celery_app/tasks.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем синтаксис
    if not test_file_syntax(file_path):
        return False
    
    # Проверяем структуру
    required_elements = [
        "def train_trade_aggregator_task",
        "def evaluate_trade_aggregator_task",
        "TradeAggregatorService"
    ]
    
    return test_file_structure(file_path, required_elements)

def test_api_endpoints():
    """Тест API endpoints"""
    file_path = "src/backend/app/routers/apidb_agent/router.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем синтаксис
    if not test_file_syntax(file_path):
        return False
    
    # Проверяем структуру
    required_elements = [
        "def train_trade_aggregator",
        "def evaluate_trade_aggregator",
        "def get_trade_aggregator_models",
        "def predict_trade_aggregator",
        "TradeAggregatorService"
    ]
    
    return test_file_structure(file_path, required_elements)

def test_schemas():
    """Тест Pydantic схем"""
    file_path = "src/backend/app/schemas/agent.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден")
        return False
    
    # Проверяем синтаксис
    if not test_file_syntax(file_path):
        return False
    
    # Проверяем структуру
    required_elements = [
        "class TradeAggregatorTrainConfig",
        "class TradeAggregatorModel",
        "class TradeAggregatorPrediction",
        "TRADE_AGGREGATOR = \"Trade_aggregator\""
    ]
    
    return test_file_structure(file_path, required_elements)

def test_frontend_components():
    """Тест фронтенд компонентов"""
    components = [
        ("frontend/src/components/profile/TradeAggregatorTrainPanel.jsx", [
            "TradeAggregatorTrainPanel",
            "useState",
            "useEffect",
            "mode",
            "weights",
            "thresholds",
            "risk_limits",
            "portfolio"
        ]),
        ("frontend/src/components/profile/TrainAgentModal.jsx", [
            "TradeAggregatorTrainPanel",
            "tradeAggregatorConfig",
            "isAgentTradeAggregator"
        ]),
        ("frontend/src/services/mlService.js", [
            "tradeAggregator: {",
            "train:",
            "evaluate:",
            "getModels:",
            "predict:"
        ])
    ]
    
    results = []
    for file_path, required_elements in components:
        if not os.path.exists(file_path):
            print(f"❌ Файл {file_path} не найден")
            results.append(False)
            continue
        
        # Для JS/JSX файлов проверяем только структуру
        if file_path.endswith(('.js', '.jsx')):
            result = test_file_structure(file_path, required_elements)
        else:
            # Для Python файлов проверяем синтаксис и структуру
            if not test_file_syntax(file_path):
                result = False
            else:
                result = test_file_structure(file_path, required_elements)
        
        results.append(result)
    
    return all(results)

def main():
    """Основная функция"""
    print("🧪 Базовые тесты для всех модулей...")
    
    tests = [
        ("Trade Aggregator Service", test_trade_aggregator_service),
        ("Celery Tasks", test_celery_tasks),
        ("API Endpoints", test_api_endpoints),
        ("Pydantic Schemas", test_schemas),
        ("Frontend Components", test_frontend_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Тестируем {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} прошел успешно")
        else:
            print(f"❌ {test_name} провалился")
    
    print(f"\n📊 Результат: {passed}/{total} модулей прошли тесты")
    
    if passed == total:
        print("🎉 Все модули прошли базовые тесты успешно!")
        return 0
    else:
        print("⚠️ Некоторые модули не прошли тесты")
        return 1

if __name__ == '__main__':
    sys.exit(main())
