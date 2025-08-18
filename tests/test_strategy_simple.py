#!/usr/bin/env python3
"""
Упрощенные тесты для Strategy модуля без PyTorch зависимостей
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestStrategySimple(unittest.TestCase):
    """Упрощенные тесты для Strategy модуля"""
    
    def setUp(self):
        """Настройка тестов"""
        # Патчим импорт PyTorch чтобы избежать конфликтов
        with patch.dict('sys.modules', {'torch': Mock()}):
            try:
                # Проверяем наличие файлов strategy
                self.strategy_files = [
                    'src/backend/app/routers/strategy/router.py'
                ]
            except ImportError as e:
                self.skipTest(f"Strategy модули недоступны: {e}")
    
    def test_strategy_files_exist(self):
        """Тест существования файлов strategy"""
        for file_path in self.strategy_files:
            self.assertTrue(os.path.exists(file_path), f"Файл {file_path} не найден")
    
    def test_strategy_router_structure(self):
        """Тест структуры strategy router"""
        router_file = 'src/backend/app/routers/strategy/router.py'
        if os.path.exists(router_file):
            with open(router_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем наличие основных элементов
            self.assertIn('from fastapi import', content)
            self.assertIn('router = APIRouter', content)
            self.assertIn('@router.post', content)
            self.assertIn('@router.get', content)
            self.assertIn('@router.delete', content)
    
    def test_strategy_endpoints_exist(self):
        """Тест наличия основных endpoints strategy"""
        router_file = 'src/backend/app/routers/strategy/router.py'
        if os.path.exists(router_file):
            with open(router_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем наличие основных endpoints
            required_endpoints = [
                '/create',
                '/list',
                '/{strategy_id}'
            ]
            
            for endpoint in required_endpoints:
                self.assertIn(endpoint, content, f"Endpoint {endpoint} не найден")
    
    def test_strategy_imports_exist(self):
        """Тест наличия основных импортов strategy"""
        router_file = 'src/backend/app/routers/strategy/router.py'
        if os.path.exists(router_file):
            with open(router_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем наличие основных импортов
            required_imports = [
                'from fastapi import',
                'from backend.app.configuration import',
                'from core.database.orm import'
            ]
            
            for import_line in required_imports:
                self.assertIn(import_line, content, f"Импорт {import_line} не найден")
    
    def test_strategy_schemas_exist(self):
        """Тест наличия схем strategy"""
        # Проверяем наличие файлов схем
        schema_files = [
            'src/backend/app/configuration/schemas/strategy.py'
        ]
        
        for schema_file in schema_files:
            if os.path.exists(schema_file):
                with open(schema_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Проверяем наличие классов схем
                self.assertIn('class Strategy', content)
                self.assertIn('class StrategyCreate', content)
                self.assertIn('class StrategyResponse', content)
    
    def test_strategy_orm_exist(self):
        """Тест наличия ORM моделей strategy"""
        orm_file = 'src/core/database/orm/strategy.py'
        if os.path.exists(orm_file):
            with open(orm_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем наличие основных функций
            required_functions = [
                'orm_create_strategy',
                'orm_get_strategy',
                'orm_get_strategies',
                'orm_delete_strategy'
            ]
            
            for func_name in required_functions:
                self.assertIn(f'def {func_name}', content, f"Функция {func_name} не найдена")

def main():
    """Запуск тестов"""
    print("🧪 Запуск упрощенных тестов Strategy модуля...")
    
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    loader = unittest.TestLoader()
    test_suite.addTest(loader.loadTestsFromTestCase(TestStrategySimple))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Выводим результат
    print(f"\nТесты завершены: {result.testsRun} тестов выполнено")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Провалов: {len(result.failures)}")
    
    if result.wasSuccessful():
        print("✅ Все тесты прошли успешно!")
        return 0
    else:
        print("❌ Некоторые тесты провалились")
        return 1

if __name__ == '__main__':
    exit_code = main()
    assert exit_code == 0, "Strategy simple tests failed"
