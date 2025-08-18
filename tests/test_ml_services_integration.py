"""
Интеграционные тесты для ML сервисов
"""
import sys
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta

def test_news_background_service_integration():
    """Тест интеграции News Background Service"""
    print("🧪 Testing News Background Service integration...")
    
    try:
        # Проверяем существование сервиса
        service_file = 'src/core/services/news_background_service.py'
        if not os.path.exists(service_file):
            print(f"❌ Service file missing: {service_file}")
            return False
        
        print(f"✅ Service file exists: {service_file}")
        
        # Проверяем структуру сервиса
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_methods = [
            'class NewsBackgroundService',
            'def process_news(',
            'def calculate_background(',
            'def get_background(',
            'def update_background('
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✅ Found method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        # Проверяем импорты
        required_imports = [
            'from ..database.orm.news import',
            'from ..database.orm.market import',
            'import redis'
        ]
        
        for imp in required_imports:
            if imp in content:
                print(f"✅ Found import: {imp}")
            else:
                print(f"❌ Missing import: {imp}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing News Background Service: {e}")
        return False


def test_pred_time_service_integration():
    """Тест интеграции Pred Time Service"""
    print("🧪 Testing Pred Time Service integration...")
    
    try:
        # Проверяем существование сервиса
        service_file = 'src/core/services/pred_time_service.py'
        if not os.path.exists(service_file):
            print(f"❌ Service file missing: {service_file}")
            return False
        
        print(f"✅ Service file exists: {service_file}")
        
        # Проверяем структуру сервиса
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_methods = [
            'class PredTimeService',
            'def train_model(',
            'def predict(',
            'def evaluate_model(',
            'def save_model(',
            'def load_model('
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✅ Found method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        # Проверяем импорты ML библиотек
        ml_imports = [
            'import torch',
            'import numpy as np',
            'import pandas as pd',
            'from sklearn.preprocessing import'
        ]
        
        found_ml_imports = 0
        for imp in ml_imports:
            if imp in content:
                print(f"✅ Found ML import: {imp}")
                found_ml_imports += 1
        
        if found_ml_imports >= 2:
            print("✅ ML libraries properly imported")
        else:
            print("⚠️  Limited ML library imports found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Pred Time Service: {e}")
        return False


def test_trade_time_service_integration():
    """Тест интеграции Trade Time Service"""
    print("🧪 Testing Trade Time Service integration...")
    
    try:
        # Проверяем существование сервиса
        service_file = 'src/core/services/trade_time_service.py'
        if not os.path.exists(service_file):
            print(f"❌ Service file missing: {service_file}")
            return False
        
        print(f"✅ Service file exists: {service_file}")
        
        # Проверяем структуру сервиса
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_methods = [
            'class TradeTimeService',
            'def train_model(',
            'def predict(',
            'def evaluate_model(',
            'def save_model(',
            'def load_model('
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✅ Found method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        # Проверяем импорты классификаторов
        classifier_imports = [
            'import lightgbm',
            'import catboost',
            'from sklearn.ensemble import'
        ]
        
        found_classifier_imports = 0
        for imp in classifier_imports:
            if imp in content:
                print(f"✅ Found classifier import: {imp}")
                found_classifier_imports += 1
        
        if found_classifier_imports >= 1:
            print("✅ Classifier libraries properly imported")
        else:
            print("⚠️  Limited classifier library imports found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Trade Time Service: {e}")
        return False


def test_risk_service_integration():
    """Тест интеграции Risk Service"""
    print("🧪 Testing Risk Service integration...")
    
    try:
        # Проверяем существование сервиса
        service_file = 'src/core/services/risk_service.py'
        if not os.path.exists(service_file):
            print(f"❌ Service file missing: {service_file}")
            return False
        
        print(f"✅ Service file exists: {service_file}")
        
        # Проверяем структуру сервиса
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_methods = [
            'class RiskService',
            'def calculate_risk(',
            'def assess_position_size(',
            'def evaluate_portfolio_risk(',
            'def get_risk_metrics('
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✅ Found method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        # Проверяем импорты риск-метрик
        risk_imports = [
            'import numpy as np',
            'import pandas as pd',
            'from ..utils.metrics import'
        ]
        
        for imp in risk_imports:
            if imp in content:
                print(f"✅ Found risk import: {imp}")
            else:
                print(f"❌ Missing risk import: {imp}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Risk Service: {e}")
        return False


def test_trade_aggregator_service_integration():
    """Тест интеграции Trade Aggregator Service"""
    print("🧪 Testing Trade Aggregator Service integration...")
    
    try:
        # Проверяем существование сервиса
        service_file = 'src/core/services/trade_aggregator_service.py'
        if not os.path.exists(service_file):
            print(f"❌ Service file missing: {service_file}")
            return False
        
        print(f"✅ Service file exists: {service_file}")
        
        # Проверяем структуру сервиса
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_methods = [
            'class TradeAggregatorService',
            'def aggregate_signals(',
            'def calculate_portfolio_metrics(',
            'def apply_risk_management(',
            'def make_final_decision('
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✅ Found method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Trade Aggregator Service: {e}")
        return False


def test_pipeline_orchestrator_integration():
    """Тест интеграции Pipeline Orchestrator"""
    print("🧪 Testing Pipeline Orchestrator integration...")
    
    try:
        # Проверяем существование сервиса
        service_file = 'src/core/services/pipeline_orchestrator.py'
        if not os.path.exists(service_file):
            print(f"❌ Service file missing: {service_file}")
            return False
        
        print(f"✅ Service file exists: {service_file}")
        
        # Проверяем структуру сервиса
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_methods = [
            'class PipelineOrchestrator',
            'def execute_pipeline(',
            'def _load_market_data(',
            'def _process_news_background(',
            'def _run_pred_time_models(',
            'def _run_trade_time_models(',
            'def _run_risk_models(',
            'def _run_trade_aggregator(',
            'def _calculate_final_metrics('
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✅ Found method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        # Проверяем импорты всех сервисов
        service_imports = [
            'from .news_background_service import NewsBackgroundService',
            'from .pred_time_service import PredTimeService',
            'from .trade_time_service import TradeTimeService',
            'from .risk_service import RiskService',
            'from .trade_aggregator_service import TradeAggregatorService'
        ]
        
        for imp in service_imports:
            if imp in content:
                print(f"✅ Found service import: {imp}")
            else:
                print(f"❌ Missing service import: {imp}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Pipeline Orchestrator: {e}")
        return False


def test_model_versioning_integration():
    """Тест интеграции Model Versioning Service"""
    print("🧪 Testing Model Versioning Service integration...")
    
    try:
        # Проверяем существование сервиса
        service_file = 'src/core/services/model_versioning_service.py'
        if not os.path.exists(service_file):
            print(f"❌ Service file missing: {service_file}")
            return False
        
        print(f"✅ Service file exists: {service_file}")
        
        # Проверяем структуру сервиса
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_methods = [
            'class ModelVersioningService',
            'def create_version(',
            'def promote_version(',
            'def rollback_version(',
            'def list_versions(',
            'def get_version_info(',
            'def delete_version(',
            'def get_production_status(',
            'def cleanup_old_versions('
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✅ Found method: {method}")
            else:
                print(f"❌ Missing method: {method}")
                return False
        
        # Проверяем импорты ORM
        orm_imports = [
            'from ..database.orm.artifacts import',
            'from ..database.orm.agents import',
            'from ..database.engine import get_db'
        ]
        
        for imp in orm_imports:
            if imp in content:
                print(f"✅ Found ORM import: {imp}")
            else:
                print(f"❌ Missing ORM import: {imp}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Model Versioning Service: {e}")
        return False


def test_celery_tasks_integration():
    """Тест интеграции Celery задач"""
    print("🧪 Testing Celery tasks integration...")
    
    try:
        # Проверяем существование файла задач
        tasks_file = 'src/backend/celery_app/tasks.py'
        if not os.path.exists(tasks_file):
            print(f"❌ Tasks file missing: {tasks_file}")
            return False
        
        print(f"✅ Tasks file exists: {tasks_file}")
        
        # Проверяем структуру задач
        with open(tasks_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_tasks = [
            'def train_news_task(',
            'def train_pred_time_task(',
            'def train_trade_time_task(',
            'def train_risk_task(',
            'def train_trade_aggregator_task(',
            'def run_pipeline_backtest_task('
        ]
        
        for task in required_tasks:
            if task in content:
                print(f"✅ Found task: {task}")
            else:
                print(f"❌ Missing task: {task}")
                return False
        
        # Проверяем импорты сервисов в задачах
        service_imports = [
            'from core.services.news_background_service import NewsBackgroundService',
            'from core.services.pred_time_service import PredTimeService',
            'from core.services.trade_time_service import TradeTimeService',
            'from core.services.risk_service import RiskService',
            'from core.services.trade_aggregator_service import TradeAggregatorService',
            'from core.services.pipeline_orchestrator import PipelineOrchestrator'
        ]
        
        found_service_imports = 0
        for imp in service_imports:
            if imp in content:
                print(f"✅ Found service import in tasks: {imp}")
                found_service_imports += 1
        
        if found_service_imports >= 3:
            print("✅ Service imports properly integrated in tasks")
        else:
            print("⚠️  Limited service imports found in tasks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Celery tasks: {e}")
        return False


def test_api_endpoints_integration():
    """Тест интеграции API эндпоинтов"""
    print("🧪 Testing API endpoints integration...")
    
    try:
        # Проверяем существование роутера
        router_file = 'src/backend/app/routers/apidb_agent/router.py'
        if not os.path.exists(router_file):
            print(f"❌ Router file missing: {router_file}")
            return False
        
        print(f"✅ Router file exists: {router_file}")
        
        # Проверяем структуру эндпоинтов
        with open(router_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ML эндпоинты
        ml_endpoints = [
            '@router.post("/news/train")',
            '@router.post("/pred_time/train")',
            '@router.post("/trade_time/train")',
            '@router.post("/risk/train")',
            '@router.post("/trade_aggregator/train")'
        ]
        
        for endpoint in ml_endpoints:
            if endpoint in content:
                print(f"✅ Found ML endpoint: {endpoint}")
            else:
                print(f"❌ Missing ML endpoint: {endpoint}")
                return False
        
        # Pipeline эндпоинты
        pipeline_endpoints = [
            '@router.post("/pipeline/run")',
            '@router.get("/pipeline/tasks/{task_id}")',
            '@router.get("/pipeline/backtests")'
        ]
        
        for endpoint in pipeline_endpoints:
            if endpoint in content:
                print(f"✅ Found pipeline endpoint: {endpoint}")
            else:
                print(f"❌ Missing pipeline endpoint: {endpoint}")
                return False
        
        # Versioning эндпоинты
        versioning_endpoints = [
            '@router.post("/models/{agent_id}/versions")',
            '@router.post("/models/{agent_id}/versions/{version}/promote")',
            '@router.get("/models/{agent_id}/versions")'
        ]
        
        for endpoint in versioning_endpoints:
            if endpoint in content:
                print(f"✅ Found versioning endpoint: {endpoint}")
            else:
                print(f"❌ Missing versioning endpoint: {endpoint}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing API endpoints: {e}")
        return False


def test_database_integration():
    """Тест интеграции базы данных"""
    print("🧪 Testing database integration...")
    
    try:
        # Проверяем модели
        models_files = [
            'src/core/database/models/ML_models.py',
            'src/core/database/models/main_models.py'
        ]
        
        for model_file in models_files:
            if not os.path.exists(model_file):
                print(f"❌ Model file missing: {model_file}")
                return False
            print(f"✅ Model file exists: {model_file}")
        
        # Проверяем ORM методы
        orm_files = [
            'src/core/database/orm/agents.py',
            'src/core/database/orm/artifacts.py',
            'src/core/database/orm/pipelines.py',
            'src/core/database/orm/market.py',
            'src/core/database/orm/news.py'
        ]
        
        for orm_file in orm_files:
            if not os.path.exists(orm_file):
                print(f"❌ ORM file missing: {orm_file}")
                return False
            print(f"✅ ORM file exists: {orm_file}")
        
        # Проверяем миграции
        migrations_dir = 'src/core/alembic/versions'
        if os.path.exists(migrations_dir):
            migration_files = os.listdir(migrations_dir)
            if len(migration_files) >= 5:
                print(f"✅ Found {len(migration_files)} migration files")
            else:
                print(f"⚠️  Limited migration files found: {len(migration_files)}")
        else:
            print("⚠️  Migrations directory not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing database integration: {e}")
        return False


def test_frontend_integration():
    """Тест интеграции Frontend компонентов"""
    print("🧪 Testing frontend integration...")
    
    try:
        # Проверяем сервисы
        service_files = [
            'frontend/src/services/mlService.js',
            'frontend/src/services/pipelineService.js',
            'frontend/src/services/versioningService.js'
        ]
        
        for service_file in service_files:
            if not os.path.exists(service_file):
                print(f"❌ Service file missing: {service_file}")
                return False
            print(f"✅ Service file exists: {service_file}")
        
        # Проверяем компоненты
        component_files = [
            'frontend/src/components/profile/TrainAgentModal.jsx',
            'frontend/src/components/profile/ModuleTester.jsx',
            'frontend/src/components/profile/ModelVersioningPanel.jsx'
        ]
        
        for component_file in component_files:
            if not os.path.exists(component_file):
                print(f"❌ Component file missing: {component_file}")
                return False
            print(f"✅ Component file exists: {component_file}")
        
        # Проверяем интеграцию сервисов в компонентах
        train_modal_file = 'frontend/src/components/profile/TrainAgentModal.jsx'
        if os.path.exists(train_modal_file):
            with open(train_modal_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'import mlService' in content:
                print("✅ ML service integrated in TrainAgentModal")
            else:
                print("❌ ML service not integrated in TrainAgentModal")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing frontend integration: {e}")
        return False


def main():
    """Основная функция тестирования"""
    print("🚀 Starting ML Services Integration Tests...")
    
    tests = [
        test_news_background_service_integration,
        test_pred_time_service_integration,
        test_trade_time_service_integration,
        test_risk_service_integration,
        test_trade_aggregator_service_integration,
        test_pipeline_orchestrator_integration,
        test_model_versioning_integration,
        test_celery_tasks_integration,
        test_api_endpoints_integration,
        test_database_integration,
        test_frontend_integration
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
    
    print(f"📊 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️  Some integration tests failed!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
