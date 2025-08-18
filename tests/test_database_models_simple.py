"""
Упрощенные тесты для новых таблиц БД без импорта моделей
"""
import sys
import os
import ast

def test_file_structure():
    """Тест структуры файлов БД"""
    print("🧪 Testing database file structure...")
    
    # Проверяем существование файлов
    files_to_check = [
        'src/core/database/models/main_models.py',
        'src/core/database/models/ML_models.py',
        'src/core/utils/metrics.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            return False
    
    return True


def test_main_models_content():
    """Тест содержимого main_models.py"""
    print("\n🔍 Testing main_models.py content...")
    
    try:
        with open('src/core/database/models/main_models.py', 'r') as f:
            content = f.read()
        
        # Проверяем наличие классов
        classes_to_find = [
            'class Artifact',
            'class Backtest', 
            'class Pipeline',
            'class NewsBackground'
        ]
        
        for class_name in classes_to_find:
            if class_name in content:
                print(f"✅ Found: {class_name}")
            else:
                print(f"❌ Missing: {class_name}")
                return False
        
        # Проверяем синтаксис
        try:
            ast.parse(content)
            print("✅ Syntax is valid")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading main_models.py: {e}")
        return False


def test_ml_models_content():
    """Тест содержимого ML_models.py"""
    print("\n🔍 Testing ML_models.py content...")
    
    try:
        with open('src/core/database/models/ML_models.py', 'r') as f:
            content = f.read()
        
        # Проверяем наличие классов и связей
        items_to_find = [
            'class Agent',
            'artifacts: Mapped[list[\'Artifact\']]',
            'from .main_models import Artifact'
        ]
        
        for item in items_to_find:
            if item in content:
                print(f"✅ Found: {item}")
            else:
                print(f"❌ Missing: {item}")
                return False
        
        # Проверяем синтаксис
        try:
            ast.parse(content)
            print("✅ Syntax is valid")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading ML_models.py: {e}")
        return False


def test_metrics_content():
    """Тест содержимого metrics.py"""
    print("\n🔍 Testing metrics.py content...")
    
    try:
        with open('src/core/utils/metrics.py', 'r') as f:
            content = f.read()
        
        # Проверяем наличие функций
        functions_to_find = [
            'def calculate_regression_metrics',
            'def calculate_classification_metrics',
            'def calculate_risk_metrics',
            'def calculate_trading_metrics',
            'def calculate_portfolio_metrics'
        ]
        
        for func_name in functions_to_find:
            if func_name in content:
                print(f"✅ Found: {func_name}")
            else:
                print(f"❌ Missing: {func_name}")
                return False
        
        # Проверяем синтаксис
        try:
            ast.parse(content)
            print("✅ Syntax is valid")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading metrics.py: {e}")
        return False


def test_migration_files():
    """Тест файлов миграций"""
    print("\n🔍 Testing migration files...")
    
    migration_files = [
        'src/core/alembic/versions/2025_06_22_1200-add-pipeline-table.py',
        'src/core/alembic/versions/2025_06_22_1800-add-backtests-table.py',
        'src/core/alembic/versions/2025_06_22_1900-add-news-background-table.py',
        'src/core/alembic/versions/2025_06_22_2000-add-artifacts-table.py'
    ]
    
    for file_path in migration_files:
        if os.path.exists(file_path):
            print(f"✅ Migration exists: {os.path.basename(file_path)}")
        else:
            print(f"❌ Migration missing: {os.path.basename(file_path)}")
            return False
    
    return True


def main():
    """Запуск всех тестов"""
    print("🧪 Running simplified database structure tests...")
    
    tests = [
        test_file_structure,
        test_main_models_content,
        test_ml_models_content,
        test_metrics_content,
        test_migration_files
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
        print("🎉 All database structure tests passed!")
        return 0
    else:
        print("⚠️ Some database structure tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
