"""
Тесты для Pipeline Orchestrator
"""
import sys
import os
import ast
import json
from datetime import datetime

def test_pipeline_orchestrator_structure():
    """Тест структуры Pipeline Orchestrator"""
    print("🧪 Testing Pipeline Orchestrator structure...")
    
    # Проверяем существование файла
    file_path = 'src/core/services/pipeline_orchestrator.py'
    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    # Читаем файл и проверяем структуру
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие ключевых элементов
    required_elements = [
        'class PipelineOrchestrator',
        'def execute_pipeline',
        'def _load_market_data',
        'def _process_news_background',
        'def _run_pred_time_models',
        'def _run_trade_time_models',
        'def _run_risk_models',
        'def _run_trade_aggregator',
        'def _calculate_final_metrics',
        'def _save_artifacts'
    ]
    
    for element in required_elements:
        if element in content:
            print(f"✅ Found: {element}")
        else:
            print(f"❌ Missing: {element}")
            return False
    
    # Проверяем синтаксис Python
    try:
        ast.parse(content)
        print("✅ Python syntax is valid")
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    
    return True


def test_celery_task_structure():
    """Тест структуры Celery задачи"""
    print("🧪 Testing Celery task structure...")
    
    file_path = 'src/backend/celery_app/tasks.py'
    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие новой задачи
    if 'def run_pipeline_backtest_task(' in content:
        print("✅ Found: run_pipeline_backtest_task function")
    else:
        print("❌ Missing: run_pipeline_backtest_task function")
        return False
    
    # Проверяем импорты
    required_imports = [
        'from core.services.pipeline_orchestrator import PipelineOrchestrator',
        'from core.database.engine import get_db'
    ]
    
    for imp in required_imports:
        if imp in content:
            print(f"✅ Found import: {imp}")
        else:
            print(f"❌ Missing import: {imp}")
            return False
    
    return True


def test_api_endpoints_structure():
    """Тест структуры API эндпоинтов"""
    print("🧪 Testing API endpoints structure...")
    
    file_path = 'src/backend/app/routers/apidb_agent/router.py'
    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие новых эндпоинтов
    required_endpoints = [
        '@router.post("/pipeline/run")',
        '@router.get("/pipeline/tasks/{task_id}")',
        '@router.post("/pipeline/tasks/{task_id}/revoke")',
        '@router.get("/pipeline/backtests")',
        '@router.get("/pipeline/backtests/{backtest_id}")',
        '@router.get("/pipeline/artifacts/{path:path}")'
    ]
    
    for endpoint in required_endpoints:
        if endpoint in content:
            print(f"✅ Found endpoint: {endpoint}")
        else:
            print(f"❌ Missing endpoint: {endpoint}")
            return False
    
    return True


def test_frontend_service_structure():
    """Тест структуры Frontend сервиса"""
    print("🧪 Testing Frontend service structure...")
    
    file_path = 'frontend/src/services/pipelineService.js'
    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие методов
    required_methods = [
        'async runPipeline(',
        'async getTaskStatus(',
        'async revokeTask(',
        'async listBacktests(',
        'async getBacktest(',
        'async downloadArtifact(',
        'async pollTaskStatus(',
        'formatProgress(',
        'formatMetrics(',
        'getBacktestStatus('
    ]
    
    for method in required_methods:
        if method in content:
            print(f"✅ Found method: {method}")
        else:
            print(f"❌ Missing method: {method}")
            return False
    
    return True


def test_orm_methods_structure():
    """Тест структуры ORM методов"""
    print("🧪 Testing ORM methods structure...")
    
    # Проверяем artifacts.py
    artifacts_file = 'src/core/database/orm/artifacts.py'
    if not os.path.exists(artifacts_file):
        print(f"❌ File missing: {artifacts_file}")
        return False
    
    print(f"✅ File exists: {artifacts_file}")
    
    with open(artifacts_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем методы artifacts
    artifacts_methods = [
        'def orm_create_artifact(',
        'def orm_get_artifact_by_id(',
        'def orm_get_artifacts_by_agent(',
        'def orm_get_latest_artifact(',
        'def orm_get_artifacts_by_version(',
        'def orm_delete_artifact(',
        'def orm_delete_artifacts_by_agent(',
        'def orm_get_artifact_stats(',
        'def orm_cleanup_old_artifacts('
    ]
    
    for method in artifacts_methods:
        if method in content:
            print(f"✅ Found artifacts method: {method}")
        else:
            print(f"❌ Missing artifacts method: {method}")
            return False
    
    # Проверяем pipelines.py
    pipelines_file = 'src/core/database/orm/pipelines.py'
    if not os.path.exists(pipelines_file):
        print(f"❌ File missing: {pipelines_file}")
        return False
    
    print(f"✅ File exists: {pipelines_file}")
    
    with open(pipelines_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем методы pipelines
    pipeline_methods = [
        'def orm_create_pipeline(',
        'def orm_get_pipeline_by_id(',
        'def orm_get_pipelines(',
        'def orm_update_pipeline(',
        'def orm_delete_pipeline(',
        'def orm_create_backtest(',
        'def orm_get_backtest_by_id(',
        'def orm_get_backtests(',
        'def orm_update_backtest_status(',
        'def orm_delete_backtest(',
        'def orm_get_backtest_stats(',
        'def orm_cleanup_old_backtests('
    ]
    
    for method in pipeline_methods:
        if method in content:
            print(f"✅ Found pipeline method: {method}")
        else:
            print(f"❌ Missing pipeline method: {method}")
            return False
    
    return True


def main():
    """Основная функция тестирования"""
    print("🚀 Starting Pipeline Orchestrator tests...")
    
    tests = [
        test_pipeline_orchestrator_structure,
        test_celery_task_structure,
        test_api_endpoints_structure,
        test_frontend_service_structure,
        test_orm_methods_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} passed\n")
            else:
                print(f"❌ {test.__name__} failed\n")
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
