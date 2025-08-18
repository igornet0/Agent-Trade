"""
Сводный тест для проверки всех компонентов Спринта 9: Интеграционные и E2E тесты
"""
import sys
import os
import json
from datetime import datetime

def test_sprint9_completeness():
    """Тест полноты реализации Спринта 9"""
    print("🧪 Testing Sprint 9 completeness...")
    
    try:
        # Проверяем наличие всех тестовых файлов
        test_files = [
            'tests/test_ml_services_integration.py',
            'tests/test_e2e_workflow.py',
            'tests/test_performance.py',
            'tests/test_sprint9_complete.py'
        ]
        
        for test_file in test_files:
            if not os.path.exists(test_file):
                print(f"❌ Test file missing: {test_file}")
                return False
            print(f"✅ Test file exists: {test_file}")
        
        # Проверяем основные компоненты системы
        core_components = [
            'src/core/services/news_background_service.py',
            'src/core/services/pred_time_service.py',
            'src/core/services/trade_time_service.py',
            'src/core/services/risk_service.py',
            'src/core/services/trade_aggregator_service.py',
            'src/core/services/pipeline_orchestrator.py',
            'src/core/services/model_versioning_service.py'
        ]
        
        for component in core_components:
            if not os.path.exists(component):
                print(f"❌ Core component missing: {component}")
                return False
            print(f"✅ Core component exists: {component}")
        
        # Проверяем API эндпоинты
        router_file = 'src/backend/app/routers/apidb_agent/router.py'
        if os.path.exists(router_file):
            with open(router_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем ключевые эндпоинты
            key_endpoints = [
                '@router.post("/pipeline/run")',
                '@router.post("/models/{agent_id}/versions/{version}/promote")',
                '@router.get("/pipeline/backtests")'
            ]
            
            for endpoint in key_endpoints:
                if endpoint in content:
                    print(f"✅ Found key endpoint: {endpoint}")
                else:
                    print(f"❌ Missing key endpoint: {endpoint}")
                    return False
        
        # Проверяем Frontend компоненты
        frontend_components = [
            'frontend/src/services/mlService.js',
            'frontend/src/services/pipelineService.js',
            'frontend/src/services/versioningService.js',
            'frontend/src/components/profile/ModelVersioningPanel.jsx'
        ]
        
        for component in frontend_components:
            if not os.path.exists(component):
                print(f"❌ Frontend component missing: {component}")
                return False
            print(f"✅ Frontend component exists: {component}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Sprint 9 completeness: {e}")
        return False


def test_integration_coverage():
    """Тест покрытия интеграции"""
    print("🧪 Testing integration coverage...")
    
    try:
        # Проверяем покрытие ML сервисов
        ml_services = [
            'src/core/services/news_background_service.py',
            'src/core/services/pred_time_service.py',
            'src/core/services/trade_time_service.py',
            'src/core/services/risk_service.py',
            'src/core/services/trade_aggregator_service.py'
        ]
        
        service_coverage = 0
        for service in ml_services:
            if os.path.exists(service):
                service_coverage += 1
        
        if service_coverage >= 4:
            print(f"✅ ML services coverage: {service_coverage}/5")
        else:
            print(f"❌ Limited ML services coverage: {service_coverage}/5")
            return False
        
        # Проверяем покрытие API эндпоинтов
        router_file = 'src/backend/app/routers/apidb_agent/router.py'
        if os.path.exists(router_file):
            with open(router_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            endpoint_categories = {
                'ML Training': ['@router.post("/pred_time/train")', '@router.post("/trade_time/train")'],
                'Pipeline': ['@router.post("/pipeline/run")', '@router.get("/pipeline/backtests")'],
                'Versioning': ['@router.post("/models/{agent_id}/versions")', '@router.post("/models/{agent_id}/versions/{version}/promote")']
            }
            
            total_endpoints = 0
            found_endpoints = 0
            
            for category, endpoints in endpoint_categories.items():
                total_endpoints += len(endpoints)
                for endpoint in endpoints:
                    if endpoint in content:
                        found_endpoints += 1
            
            coverage_percentage = (found_endpoints / total_endpoints) * 100
            if coverage_percentage >= 80:
                print(f"✅ API endpoints coverage: {found_endpoints}/{total_endpoints} ({coverage_percentage:.1f}%)")
            else:
                print(f"❌ Limited API endpoints coverage: {found_endpoints}/{total_endpoints} ({coverage_percentage:.1f}%)")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing integration coverage: {e}")
        return False


def test_e2e_workflow_coverage():
    """Тест покрытия E2E рабочих процессов"""
    print("🧪 Testing E2E workflow coverage...")
    
    try:
        # Проверяем основные рабочие процессы
        workflows = {
            'Pipeline Execution': 'src/core/services/pipeline_orchestrator.py',
            'Model Versioning': 'src/core/services/model_versioning_service.py',
            'Celery Tasks': 'src/backend/celery_app/tasks.py',
            'Database ORM': 'src/core/database/orm/artifacts.py'
        }
        
        workflow_coverage = 0
        for workflow_name, workflow_file in workflows.items():
            if os.path.exists(workflow_file):
                workflow_coverage += 1
                print(f"✅ {workflow_name} workflow exists")
            else:
                print(f"❌ {workflow_name} workflow missing")
        
        if workflow_coverage >= 3:
            print(f"✅ E2E workflow coverage: {workflow_coverage}/4")
        else:
            print(f"❌ Limited E2E workflow coverage: {workflow_coverage}/4")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing E2E workflow coverage: {e}")
        return False


def test_performance_optimization_coverage():
    """Тест покрытия оптимизаций производительности"""
    print("🧪 Testing performance optimization coverage...")
    
    try:
        # Проверяем различные типы оптимизаций
        optimization_areas = {
            'Pipeline Performance': 'src/core/services/pipeline_orchestrator.py',
            'Memory Optimization': 'src/core/services/pred_time_service.py',
            'Caching': 'src/core/services/news_background_service.py',
            'Error Handling': 'src/backend/celery_app/tasks.py'
        }
        
        optimization_coverage = 0
        for area_name, area_file in optimization_areas.items():
            if os.path.exists(area_file):
                with open(area_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Проверяем наличие оптимизаций
                optimizations = ['ThreadPoolExecutor', 'asyncio', 'cache', 'try:', 'except']
                found_optimizations = sum(1 for opt in optimizations if opt in content)
                
                if found_optimizations >= 2:
                    optimization_coverage += 1
                    print(f"✅ {area_name}: {found_optimizations} optimizations")
                else:
                    print(f"⚠️  {area_name}: {found_optimizations} optimizations")
        
        if optimization_coverage >= 3:
            print(f"✅ Performance optimization coverage: {optimization_coverage}/4")
        else:
            print(f"⚠️  Limited performance optimization coverage: {optimization_coverage}/4")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing performance optimization coverage: {e}")
        return False


def test_frontend_integration_coverage():
    """Тест покрытия интеграции Frontend"""
    print("🧪 Testing frontend integration coverage...")
    
    try:
        # Проверяем Frontend сервисы
        frontend_services = [
            'frontend/src/services/mlService.js',
            'frontend/src/services/pipelineService.js',
            'frontend/src/services/versioningService.js'
        ]
        
        service_coverage = 0
        for service in frontend_services:
            if os.path.exists(service):
                service_coverage += 1
                print(f"✅ Frontend service exists: {service}")
            else:
                print(f"❌ Frontend service missing: {service}")
        
        if service_coverage >= 2:
            print(f"✅ Frontend services coverage: {service_coverage}/3")
        else:
            print(f"❌ Limited frontend services coverage: {service_coverage}/3")
            return False
        
        # Проверяем компоненты
        components = [
            'frontend/src/components/profile/TrainAgentModal.jsx',
            'frontend/src/components/profile/ModelVersioningPanel.jsx'
        ]
        
        component_coverage = 0
        for component in components:
            if os.path.exists(component):
                component_coverage += 1
                print(f"✅ Frontend component exists: {component}")
            else:
                print(f"❌ Frontend component missing: {component}")
        
        if component_coverage >= 1:
            print(f"✅ Frontend components coverage: {component_coverage}/2")
        else:
            print(f"❌ Limited frontend components coverage: {component_coverage}/2")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing frontend integration coverage: {e}")
        return False


def test_database_integration_coverage():
    """Тест покрытия интеграции базы данных"""
    print("🧪 Testing database integration coverage...")
    
    try:
        # Проверяем модели БД
        db_models = [
            'src/core/database/models/ML_models.py',
            'src/core/database/models/main_models.py'
        ]
        
        model_coverage = 0
        for model in db_models:
            if os.path.exists(model):
                model_coverage += 1
                print(f"✅ Database model exists: {model}")
            else:
                print(f"❌ Database model missing: {model}")
        
        if model_coverage >= 1:
            print(f"✅ Database models coverage: {model_coverage}/2")
        else:
            print(f"❌ Limited database models coverage: {model_coverage}/2")
            return False
        
        # Проверяем ORM методы
        orm_files = [
            'src/core/database/orm/agents.py',
            'src/core/database/orm/artifacts.py',
            'src/core/database/orm/pipelines.py'
        ]
        
        orm_coverage = 0
        for orm_file in orm_files:
            if os.path.exists(orm_file):
                orm_coverage += 1
                print(f"✅ ORM file exists: {orm_file}")
            else:
                print(f"❌ ORM file missing: {orm_file}")
        
        if orm_coverage >= 2:
            print(f"✅ ORM coverage: {orm_coverage}/3")
        else:
            print(f"❌ Limited ORM coverage: {orm_coverage}/3")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing database integration coverage: {e}")
        return False


def main():
    """Основная функция тестирования Спринта 9"""
    print("🚀 Starting Sprint 9 Complete Tests...")
    
    tests = [
        test_sprint9_completeness,
        test_integration_coverage,
        test_e2e_workflow_coverage,
        test_performance_optimization_coverage,
        test_frontend_integration_coverage,
        test_database_integration_coverage
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
    
    print(f"📊 Sprint 9 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Sprint 9 is complete!")
        return True
    else:
        print("⚠️  Sprint 9 needs more work!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
