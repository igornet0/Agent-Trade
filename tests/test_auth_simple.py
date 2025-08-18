#!/usr/bin/env python3
"""
Упрощенные тесты для Auth модуля без PyTorch зависимостей
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestAuthSimple(unittest.TestCase):
    """Упрощенные тесты для Auth модуля"""
    
    def setUp(self):
        """Настройка тестов"""
        # Патчим импорт PyTorch чтобы избежать конфликтов
        with patch.dict('sys.modules', {'torch': Mock()}):
            try:
                # Проверяем наличие файлов auth
                self.auth_files = [
                    'src/backend/app/configuration/auth.py'
                ]
            except ImportError as e:
                self.skipTest(f"Auth модули недоступны: {e}")
    
    def test_auth_files_exist(self):
        """Тест существования файлов auth"""
        for file_path in self.auth_files:
            self.assertTrue(os.path.exists(file_path), f"Файл {file_path} не найден")
    
    def test_auth_functions_exist(self):
        """Тест наличия основных функций auth"""
        config_file = 'src/backend/app/configuration/auth.py'
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем наличие основных функций
            required_functions = [
                'verify_password',
                'get_password_hash',
                'create_access_token',
                'decode_access_token',
                'create_refresh_token',
                'decode_refresh_token',
                'get_current_token_payload',
                'validate_token_type',
                'get_user_by_token_sub',
                'get_auth_user_from_token_of_type',
                'get_current_active_auth_user',
                'validate_auth_user',
                'get_current_user',
                'verify_authorization',
                'verify_authorization_admin'
            ]
            
            for func_name in required_functions:
                self.assertIn(f'def {func_name}', content, f"Функция {func_name} не найдена")
    
    def test_auth_imports_exist(self):
        """Тест наличия основных импортов auth"""
        config_file = 'src/backend/app/configuration/auth.py'
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем наличие основных импортов
            required_imports = [
                'import asyncio',
                'from typing import',
                'from sqlalchemy.ext.asyncio import AsyncSession',
                'from fastapi import',
                'from jose import',
                'from datetime import',
                'import re',
                'from core.settings import settings',
                'from core.database import User',
                'from core.database.orm.users import orm_get_user_by_login',
                'from backend.app.configuration import'
            ]
            
            for import_line in required_imports:
                self.assertIn(import_line, content, f"Импорт {import_line} не найден")
    
    def test_auth_constants_exist(self):
        """Тест наличия констант auth"""
        config_file = 'src/backend/app/configuration/auth.py'
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем наличие констант
            required_constants = [
                'TOKEN_TYPE_FIELD',
                'ACCESS_TOKEN_TYPE',
                'REFRESH_TOKEN_TYPE',
                'EMAIL_REGEX'
            ]
            
            for const_name in required_constants:
                self.assertIn(const_name, content, f"Константа {const_name} не найдена")
    
    def test_auth_utility_functions_exist(self):
        """Тест наличия утилитарных функций auth"""
        config_file = 'src/backend/app/configuration/auth.py'
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Проверяем наличие утилитарных функций
            required_utilities = [
                'is_email'
            ]
            
            for util_name in required_utilities:
                self.assertIn(f'def {util_name}', content, f"Утилитарная функция {util_name} не найдена")

def main():
    """Запуск тестов"""
    print("🧪 Запуск упрощенных тестов Auth модуля...")
    
    # Создаем тестовый набор
    test_suite = unittest.TestSuite()
    
    # Добавляем тесты
    loader = unittest.TestLoader()
    test_suite.addTest(loader.loadTestsFromTestCase(TestAuthSimple))
    
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
    assert exit_code == 0, "Auth simple tests failed"
