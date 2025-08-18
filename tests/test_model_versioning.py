"""
Тесты для системы версионирования моделей
"""
import sys
import os
import ast
import json
from datetime import datetime

def test_model_versioning_service_structure():
    """Тест структуры Model Versioning Service"""
    print("🧪 Testing Model Versioning Service structure...")
    
    # Проверяем существование файла
    file_path = 'src/core/services/model_versioning_service.py'
    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    # Читаем файл и проверяем структуру
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие ключевых элементов
    required_elements = [
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


def test_agent_model_versioning_fields():
    """Тест полей версионирования в модели Agent"""
    print("🧪 Testing Agent model versioning fields...")
    
    file_path = 'src/core/database/models/ML_models.py'
    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие полей версионирования
    required_fields = [
        'production_version: Mapped[str]',
        'production_artifacts: Mapped[dict]',
        'agent_type: Mapped[str]'
    ]
    
    for field in required_fields:
        if field in content:
            print(f"✅ Found field: {field}")
        else:
            print(f"❌ Missing field: {field}")
            return False
    
    return True


def test_api_versioning_endpoints_structure():
    """Тест структуры API эндпоинтов версионирования"""
    print("🧪 Testing API versioning endpoints structure...")
    
    file_path = 'src/backend/app/routers/apidb_agent/router.py'
    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие эндпоинтов версионирования
    required_endpoints = [
        '@router.post("/models/{agent_id}/versions")',
        '@router.post("/models/{agent_id}/versions/{version}/promote")',
        '@router.post("/models/{agent_id}/versions/{version}/rollback")',
        '@router.get("/models/{agent_id}/versions")',
        '@router.get("/models/{agent_id}/versions/{version}")',
        '@router.delete("/models/{agent_id}/versions/{version}")',
        '@router.get("/models/{agent_id}/production")',
        '@router.post("/models/{agent_id}/versions/cleanup")'
    ]
    
    for endpoint in required_endpoints:
        if endpoint in content:
            print(f"✅ Found endpoint: {endpoint}")
        else:
            print(f"❌ Missing endpoint: {endpoint}")
            return False
    
    return True


def test_frontend_versioning_service_structure():
    """Тест структуры Frontend сервиса версионирования"""
    print("🧪 Testing Frontend versioning service structure...")
    
    file_path = 'frontend/src/services/versioningService.js'
    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие методов
    required_methods = [
        'async createVersion(',
        'async promoteVersion(',
        'async rollbackVersion(',
        'async listVersions(',
        'async getVersionInfo(',
        'async deleteVersion(',
        'async getProductionStatus(',
        'async cleanupVersions(',
        'formatFileSize(',
        'formatCreatedAt(',
        'getVersionStatus(',
        'checkVersionIntegrity(',
        'compareVersions(',
        'getVersioningRecommendations('
    ]
    
    for method in required_methods:
        if method in content:
            print(f"✅ Found method: {method}")
        else:
            print(f"❌ Missing method: {method}")
            return False
    
    return True


def test_frontend_versioning_component_structure():
    """Тест структуры Frontend компонента версионирования"""
    print("🧪 Testing Frontend versioning component structure...")
    
    file_path = 'frontend/src/components/profile/ModelVersioningPanel.jsx'
    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        return False
    
    print(f"✅ File exists: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем наличие ключевых элементов React компонента
    required_elements = [
        'import React, { useState, useEffect }',
        'import versioningService',
        'const ModelVersioningPanel',
        'useState(',
        'useEffect(',
        'loadVersions',
        'loadProductionStatus',
        'handleCreateVersion',
        'handlePromoteVersion',
        'handleRollbackVersion',
        'handleDeleteVersion',
        'handleCleanupVersions',
        'return (',
        'export default ModelVersioningPanel'
    ]
    
    for element in required_elements:
        if element in content:
            print(f"✅ Found: {element}")
        else:
            print(f"❌ Missing: {element}")
            return False
    
    return True


def test_versioning_integration():
    """Тест интеграции компонентов версионирования"""
    print("🧪 Testing versioning integration...")
    
    # Проверяем импорты в сервисе
    service_file = 'src/core/services/model_versioning_service.py'
    if os.path.exists(service_file):
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем импорты ORM методов
        required_imports = [
            'from ..database.orm.artifacts import',
            'from ..database.orm.agents import',
            'from ..database.engine import get_db'
        ]
        
        for imp in required_imports:
            if imp in content:
                print(f"✅ Found import: {imp}")
            else:
                print(f"❌ Missing import: {imp}")
                return False
    
    # Проверяем импорты в API роутере
    router_file = 'src/backend/app/routers/apidb_agent/router.py'
    if os.path.exists(router_file):
        with open(router_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем импорт сервиса
        if 'from core.services.model_versioning_service import ModelVersioningService' in content:
            print("✅ Found ModelVersioningService import in router")
        else:
            print("❌ Missing ModelVersioningService import in router")
            return False
    
    return True


def main():
    """Основная функция тестирования"""
    print("🚀 Starting Model Versioning tests...")
    
    tests = [
        test_model_versioning_service_structure,
        test_agent_model_versioning_fields,
        test_api_versioning_endpoints_structure,
        test_frontend_versioning_service_structure,
        test_frontend_versioning_component_structure,
        test_versioning_integration
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
