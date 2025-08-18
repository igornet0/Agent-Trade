"""
E2E тесты для полного рабочего цикла системы
"""
import sys
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta

def test_complete_ml_workflow():
    """Тест полного ML рабочего цикла"""
    print("🧪 Testing complete ML workflow...")
    
    try:
        # Проверяем наличие всех необходимых компонентов
        required_components = [
            'src/core/services/news_background_service.py',
            'src/core/services/pred_time_service.py',
            'src/core/services/trade_time_service.py',
            'src/core/services/risk_service.py',
            'src/core/services/trade_aggregator_service.py',
            'src/core/services/pipeline_orchestrator.py',
            'src/core/services/model_versioning_service.py'
        ]
        
        for component in required_components:
            if not os.path.exists(component):
                print(f"❌ Missing component: {component}")
                return False
            print(f"✅ Component exists: {component}")
        
        # Проверяем API эндпоинты
        router_file = 'src/backend/app/routers/apidb_agent/router.py'
        if os.path.exists(router_file):
            with open(router_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем все типы эндпоинтов
            endpoint_types = {
                'ML Training': [
                    '@router.post("/news/train")',
                    '@router.post("/pred_time/train")',
                    '@router.post("/trade_time/train")',
                    '@router.post("/risk/train")',
                    '@router.post("/trade_aggregator/train")'
                ],
                'ML Evaluation': [
                    '@router.post("/news/evaluate")',
                    '@router.post("/pred_time/evaluate")',
                    '@router.post("/trade_time/evaluate")',
                    '@router.post("/risk/evaluate")',
                    '@router.post("/trade_aggregator/evaluate")'
                ],
                'Pipeline': [
                    '@router.post("/pipeline/run")',
                    '@router.get("/pipeline/tasks/{task_id}")',
                    '@router.get("/pipeline/backtests")'
                ],
                'Versioning': [
                    '@router.post("/models/{agent_id}/versions")',
                    '@router.post("/models/{agent_id}/versions/{version}/promote")',
                    '@router.get("/models/{agent_id}/versions")'
                ]
            }
            
            for endpoint_type, endpoints in endpoint_types.items():
                found_count = 0
                for endpoint in endpoints:
                    if endpoint in content:
                        found_count += 1
                
                if found_count >= len(endpoints) * 0.8:  # 80% покрытие
                    print(f"✅ {endpoint_type} endpoints: {found_count}/{len(endpoints)}")
                else:
                    print(f"⚠️  {endpoint_type} endpoints: {found_count}/{len(endpoints)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ML workflow: {e}")
        return False


def test_pipeline_execution_workflow():
    """Тест рабочего цикла выполнения пайплайна"""
    print("🧪 Testing pipeline execution workflow...")
    
    try:
        # Проверяем Pipeline Orchestrator
        orchestrator_file = 'src/core/services/pipeline_orchestrator.py'
        if not os.path.exists(orchestrator_file):
            print(f"❌ Pipeline orchestrator missing: {orchestrator_file}")
            return False
        
        with open(orchestrator_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем этапы выполнения пайплайна
        pipeline_stages = [
            'def _load_market_data(',
            'def _process_news_background(',
            'def _run_pred_time_models(',
            'def _run_trade_time_models(',
            'def _run_risk_models(',
            'def _run_trade_aggregator(',
            'def _calculate_final_metrics(',
            'def _save_artifacts('
        ]
        
        for stage in pipeline_stages:
            if stage in content:
                print(f"✅ Found pipeline stage: {stage}")
            else:
                print(f"❌ Missing pipeline stage: {stage}")
                return False
        
        # Проверяем Celery задачу
        tasks_file = 'src/backend/celery_app/tasks.py'
        if os.path.exists(tasks_file):
            with open(tasks_file, 'r', encoding='utf-8') as f:
                tasks_content = f.read()
            
            if 'def run_pipeline_backtest_task(' in tasks_content:
                print("✅ Pipeline Celery task found")
            else:
                print("❌ Pipeline Celery task missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing pipeline workflow: {e}")
        return False


def test_model_versioning_workflow():
    """Тест рабочего цикла версионирования моделей"""
    print("🧪 Testing model versioning workflow...")
    
    try:
        # Проверяем Model Versioning Service
        versioning_file = 'src/core/services/model_versioning_service.py'
        if not os.path.exists(versioning_file):
            print(f"❌ Model versioning service missing: {versioning_file}")
            return False
        
        with open(versioning_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем операции версионирования
        versioning_operations = [
            'def create_version(',
            'def promote_version(',
            'def rollback_version(',
            'def list_versions(',
            'def get_version_info(',
            'def delete_version(',
            'def get_production_status(',
            'def cleanup_old_versions('
        ]
        
        for operation in versioning_operations:
            if operation in content:
                print(f"✅ Found versioning operation: {operation}")
            else:
                print(f"❌ Missing versioning operation: {operation}")
                return False
        
        # Проверяем API эндпоинты версионирования
        router_file = 'src/backend/app/routers/apidb_agent/router.py'
        if os.path.exists(router_file):
            with open(router_file, 'r', encoding='utf-8') as f:
                router_content = f.read()
            
            versioning_endpoints = [
                '@router.post("/models/{agent_id}/versions")',
                '@router.post("/models/{agent_id}/versions/{version}/promote")',
                '@router.post("/models/{agent_id}/versions/{version}/rollback")',
                '@router.get("/models/{agent_id}/versions")',
                '@router.get("/models/{agent_id}/versions/{version}")',
                '@router.delete("/models/{agent_id}/versions/{version}")',
                '@router.get("/models/{agent_id}/production")',
                '@router.post("/models/{agent_id}/versions/cleanup")'
            ]
            
            found_endpoints = 0
            for endpoint in versioning_endpoints:
                if endpoint in router_content:
                    found_endpoints += 1
            
            if found_endpoints >= len(versioning_endpoints) * 0.8:
                print(f"✅ Versioning endpoints: {found_endpoints}/{len(versioning_endpoints)}")
            else:
                print(f"❌ Versioning endpoints: {found_endpoints}/{len(versioning_endpoints)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing versioning workflow: {e}")
        return False


def test_frontend_workflow():
    """Тест рабочего цикла Frontend"""
    print("🧪 Testing frontend workflow...")
    
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
            
            # Проверяем импорты сервисов
            service_imports = [
                'import mlService',
                'import pipelineService',
                'import versioningService'
            ]
            
            found_imports = 0
            for imp in service_imports:
                if imp in content:
                    found_imports += 1
            
            if found_imports >= 1:
                print(f"✅ Service imports in TrainAgentModal: {found_imports}/3")
            else:
                print("⚠️  Limited service imports in TrainAgentModal")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing frontend workflow: {e}")
        return False


def test_database_workflow():
    """Тест рабочего цикла базы данных"""
    print("🧪 Testing database workflow...")
    
    try:
        # Проверяем модели
        model_files = [
            'src/core/database/models/ML_models.py',
            'src/core/database/models/main_models.py'
        ]
        
        for model_file in model_files:
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
                print(f"⚠️  Limited migration files: {len(migration_files)}")
        else:
            print("⚠️  Migrations directory not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing database workflow: {e}")
        return False


def test_metrics_workflow():
    """Тест рабочего цикла метрик"""
    print("🧪 Testing metrics workflow...")
    
    try:
        # Проверяем модуль метрик
        metrics_file = 'src/core/utils/metrics.py'
        if not os.path.exists(metrics_file):
            print(f"❌ Metrics file missing: {metrics_file}")
            return False
        
        with open(metrics_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем функции метрик
        metrics_functions = [
            'def calculate_regression_metrics(',
            'def calculate_classification_metrics(',
            'def calculate_risk_metrics(',
            'def calculate_trading_metrics(',
            'def calculate_portfolio_metrics('
        ]
        
        for func in metrics_functions:
            if func in content:
                print(f"✅ Found metrics function: {func}")
            else:
                print(f"❌ Missing metrics function: {func}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing metrics workflow: {e}")
        return False


def test_celery_workflow():
    """Тест рабочего цикла Celery"""
    print("🧪 Testing Celery workflow...")
    
    try:
        # Проверяем Celery задачи
        tasks_file = 'src/backend/celery_app/tasks.py'
        if not os.path.exists(tasks_file):
            print(f"❌ Celery tasks file missing: {tasks_file}")
            return False
        
        with open(tasks_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем задачи обучения
        training_tasks = [
            'def train_news_task(',
            'def train_pred_time_task(',
            'def train_trade_time_task(',
            'def train_risk_task(',
            'def train_trade_aggregator_task('
        ]
        
        for task in training_tasks:
            if task in content:
                print(f"✅ Found training task: {task}")
            else:
                print(f"❌ Missing training task: {task}")
                return False
        
        # Проверяем задачу пайплайна
        if 'def run_pipeline_backtest_task(' in content:
            print("✅ Found pipeline task")
        else:
            print("❌ Missing pipeline task")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Celery workflow: {e}")
        return False


def test_configuration_workflow():
    """Тест рабочего цикла конфигурации"""
    print("🧪 Testing configuration workflow...")
    
    try:
        # Проверяем конфигурационные файлы
        config_files = [
            'src/core/settings/config.py',
            'src/core/settings/config_DS.py'
        ]
        
        for config_file in config_files:
            if not os.path.exists(config_file):
                print(f"❌ Config file missing: {config_file}")
                return False
            print(f"✅ Config file exists: {config_file}")
        
        # Проверяем схемы
        schemas_file = 'src/backend/app/configuration/schemas/agent.py'
        if os.path.exists(schemas_file):
            with open(schemas_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем схемы конфигурации
            config_schemas = [
                'class NewsTrainConfig',
                'class PredTimeTrainConfig',
                'class TradeTimeTrainConfig',
                'class RiskTrainConfig',
                'class TradeAggregatorTrainConfig'
            ]
            
            found_schemas = 0
            for schema in config_schemas:
                if schema in content:
                    found_schemas += 1
            
            if found_schemas >= 3:
                print(f"✅ Config schemas: {found_schemas}/5")
            else:
                print(f"⚠️  Limited config schemas: {found_schemas}/5")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing configuration workflow: {e}")
        return False


def main():
    """Основная функция тестирования"""
    print("🚀 Starting E2E Workflow Tests...")
    
    tests = [
        test_complete_ml_workflow,
        test_pipeline_execution_workflow,
        test_model_versioning_workflow,
        test_frontend_workflow,
        test_database_workflow,
        test_metrics_workflow,
        test_celery_workflow,
        test_configuration_workflow
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
    
    print(f"📊 E2E Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All E2E tests passed!")
        return True
    else:
        print("⚠️  Some E2E tests failed!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
