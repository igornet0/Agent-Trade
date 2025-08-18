"""
Тесты для новых ORM методов
"""
import sys
import os
import ast

def test_artifacts_orm_structure():
    """Тест структуры ORM методов для артефактов"""
    print("🔍 Testing artifacts ORM methods structure...")
    
    try:
        with open('src/core/database/orm/artifacts.py', 'r') as f:
            content = f.read()
        
        # Проверяем наличие функций
        functions_to_find = [
            'def orm_create_artifact',
            'def orm_get_artifact_by_id',
            'def orm_get_artifacts_by_agent',
            'def orm_get_latest_artifact',
            'def orm_get_artifacts_by_version',
            'def orm_delete_artifact',
            'def orm_delete_artifacts_by_agent',
            'def orm_get_artifact_stats',
            'def orm_cleanup_old_artifacts'
        ]
        
        for func_name in functions_to_find:
            if func_name in content:
                print(f"✅ Found: {func_name}")
            else:
                print(f"❌ Missing: {func_name}")
                return False
        
        # Проверяем импорты
        imports_to_find = [
            'from core.database.models.main_models import Artifact',
            'from core.database.models.ML_models import Agent'
        ]
        
        for import_name in imports_to_find:
            if import_name in content:
                print(f"✅ Found import: {import_name}")
            else:
                print(f"❌ Missing import: {import_name}")
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
        print(f"❌ Error reading artifacts.py: {e}")
        return False


def test_pipelines_orm_structure():
    """Тест структуры ORM методов для пайплайнов"""
    print("\n🔍 Testing pipelines ORM methods structure...")
    
    try:
        with open('src/core/database/orm/pipelines.py', 'r') as f:
            content = f.read()
        
        # Проверяем наличие функций для пайплайнов
        pipeline_functions = [
            'def orm_create_pipeline',
            'def orm_get_pipeline_by_id',
            'def orm_get_pipelines',
            'def orm_update_pipeline',
            'def orm_delete_pipeline'
        ]
        
        for func_name in pipeline_functions:
            if func_name in content:
                print(f"✅ Found: {func_name}")
            else:
                print(f"❌ Missing: {func_name}")
                return False
        
        # Проверяем наличие функций для бэктестов
        backtest_functions = [
            'def orm_create_backtest',
            'def orm_get_backtest_by_id',
            'def orm_get_backtests',
            'def orm_update_backtest_status',
            'def orm_delete_backtest',
            'def orm_get_backtest_stats',
            'def orm_cleanup_old_backtests'
        ]
        
        for func_name in backtest_functions:
            if func_name in content:
                print(f"✅ Found: {func_name}")
            else:
                print(f"❌ Missing: {func_name}")
                return False
        
        # Проверяем импорты
        imports_to_find = [
            'from core.database.models.main_models import Pipeline, Backtest',
            'from core.database.models.ML_models import Agent'
        ]
        
        for import_name in imports_to_find:
            if import_name in content:
                print(f"✅ Found import: {import_name}")
            else:
                print(f"❌ Missing import: {import_name}")
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
        print(f"❌ Error reading pipelines.py: {e}")
        return False


def test_orm_files_exist():
    """Тест существования файлов ORM"""
    print("🔍 Testing ORM files existence...")
    
    orm_files = [
        'src/core/database/orm/artifacts.py',
        'src/core/database/orm/pipelines.py'
    ]
    
    for file_path in orm_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            return False
    
    return True


def test_orm_directory_structure():
    """Тест структуры директории ORM"""
    print("\n🔍 Testing ORM directory structure...")
    
    orm_dir = 'src/core/database/orm'
    
    if os.path.exists(orm_dir):
        print(f"✅ ORM directory exists: {orm_dir}")
        
        # Проверяем содержимое директории
        files = os.listdir(orm_dir)
        print(f"✅ ORM directory contains {len(files)} files")
        
        # Проверяем наличие __init__.py
        if '__init__.py' in files:
            print("✅ __init__.py exists in ORM directory")
        else:
            print("⚠️ __init__.py missing in ORM directory")
        
        return True
    else:
        print(f"❌ ORM directory missing: {orm_dir}")
        return False


def main():
    """Запуск всех тестов"""
    print("🧪 Running ORM methods tests...")
    
    tests = [
        test_orm_files_exist,
        test_orm_directory_structure,
        test_artifacts_orm_structure,
        test_pipelines_orm_structure
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
        print("🎉 All ORM methods tests passed!")
        return 0
    else:
        print("⚠️ Some ORM methods tests failed")
        return 1


if __name__ == '__main__':
    success = main()
    assert success, "ORM methods tests failed"
