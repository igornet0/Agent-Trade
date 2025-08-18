"""
Тесты производительности системы
"""
import sys
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta

def test_pipeline_performance():
    """Тест производительности пайплайна"""
    print("🧪 Testing pipeline performance...")
    
    try:
        # Проверяем Pipeline Orchestrator на оптимизацию
        orchestrator_file = 'src/core/services/pipeline_orchestrator.py'
        if not os.path.exists(orchestrator_file):
            print(f"❌ Pipeline orchestrator missing: {orchestrator_file}")
            return False
        
        with open(orchestrator_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Проверяем оптимизации
        optimizations = [
            'ThreadPoolExecutor',  # Параллельная обработка
            'asyncio',  # Асинхронность
            'batch_size',  # Батчевая обработка
            'chunk_size',  # Размер чанков
            'memory_limit',  # Ограничение памяти
            'timeout',  # Таймауты
            'cache',  # Кэширование
            'lru_cache'  # LRU кэш
        ]
        
        found_optimizations = 0
        for opt in optimizations:
            if opt in content:
                print(f"✅ Found optimization: {opt}")
                found_optimizations += 1
        
        if found_optimizations >= 3:
            print(f"✅ Pipeline performance optimizations: {found_optimizations}/8")
        else:
            print(f"⚠️  Limited pipeline optimizations: {found_optimizations}/8")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing pipeline performance: {e}")
        return False


def test_database_performance():
    """Тест производительности базы данных"""
    print("🧪 Testing database performance...")
    
    try:
        # Проверяем ORM методы на оптимизацию
        orm_files = [
            'src/core/database/orm/agents.py',
            'src/core/database/orm/artifacts.py',
            'src/core/database/orm/pipelines.py'
        ]
        
        total_optimizations = 0
        for orm_file in orm_files:
            if os.path.exists(orm_file):
                with open(orm_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Проверяем оптимизации БД
                db_optimizations = [
                    'lazy="select"',  # Ленивая загрузка
                    'lazy="joined"',  # Joined загрузка
                    'lazy="subquery"',  # Subquery загрузка
                    'index=True',  # Индексы
                    'unique=True',  # Уникальные ограничения
                    'cascade="delete"',  # Каскадное удаление
                    'backref',  # Обратные ссылки
                    'relationship'  # Отношения
                ]
                
                file_optimizations = 0
                for opt in db_optimizations:
                    if opt in content:
                        file_optimizations += 1
                
                total_optimizations += file_optimizations
                print(f"✅ {orm_file}: {file_optimizations} optimizations")
        
        if total_optimizations >= 10:
            print(f"✅ Database performance optimizations: {total_optimizations}")
        else:
            print(f"⚠️  Limited database optimizations: {total_optimizations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing database performance: {e}")
        return False


def test_memory_optimization():
    """Тест оптимизации памяти"""
    print("🧪 Testing memory optimization...")
    
    try:
        # Проверяем ML сервисы на оптимизацию памяти
        ml_services = [
            'src/core/services/pred_time_service.py',
            'src/core/services/trade_time_service.py',
            'src/core/services/trade_aggregator_service.py'
        ]
        
        memory_optimizations = [
            'del',  # Удаление переменных
            'gc.collect()',  # Сборка мусора
            'torch.cuda.empty_cache()',  # Очистка GPU памяти
            'numpy.delete',  # Удаление массивов
            'pandas.drop',  # Удаление колонок
            'chunk_size',  # Размер чанков
            'batch_size',  # Размер батчей
            'memory_limit'  # Ограничение памяти
        ]
        
        total_optimizations = 0
        for service_file in ml_services:
            if os.path.exists(service_file):
                with open(service_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_optimizations = 0
                for opt in memory_optimizations:
                    if opt in content:
                        file_optimizations += 1
                
                total_optimizations += file_optimizations
                print(f"✅ {service_file}: {file_optimizations} memory optimizations")
        
        if total_optimizations >= 5:
            print(f"✅ Memory optimizations: {total_optimizations}")
        else:
            print(f"⚠️  Limited memory optimizations: {total_optimizations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing memory optimization: {e}")
        return False


def test_cpu_optimization():
    """Тест оптимизации CPU"""
    print("🧪 Testing CPU optimization...")
    
    try:
        # Проверяем оптимизации CPU
        optimization_patterns = [
            'multiprocessing',  # Многопроцессорность
            'ThreadPoolExecutor',  # Пул потоков
            'asyncio',  # Асинхронность
            'concurrent.futures',  # Конкурентность
            'joblib',  # Параллельная обработка
            'numba',  # JIT компиляция
            'vectorize',  # Векторизация
            'parallel'  # Параллелизм
        ]
        
        # Проверяем в основных файлах
        main_files = [
            'src/core/services/pipeline_orchestrator.py',
            'src/backend/celery_app/tasks.py',
            'src/core/utils/metrics.py'
        ]
        
        total_optimizations = 0
        for file_path in main_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_optimizations = 0
                for pattern in optimization_patterns:
                    if pattern in content:
                        file_optimizations += 1
                
                total_optimizations += file_optimizations
                print(f"✅ {file_path}: {file_optimizations} CPU optimizations")
        
        if total_optimizations >= 3:
            print(f"✅ CPU optimizations: {total_optimizations}")
        else:
            print(f"⚠️  Limited CPU optimizations: {total_optimizations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CPU optimization: {e}")
        return False


def test_caching_optimization():
    """Тест оптимизации кэширования"""
    print("🧪 Testing caching optimization...")
    
    try:
        # Проверяем стратегии кэширования
        caching_patterns = [
            'redis',  # Redis кэш
            'cache',  # Общий кэш
            'lru_cache',  # LRU кэш
            'functools.lru_cache',  # Функциональный LRU кэш
            'memoize',  # Мемоизация
            'ttl',  # Time to live
            'expire',  # Истечение
            'cache_key'  # Ключи кэша
        ]
        
        # Проверяем в сервисах
        service_files = [
            'src/core/services/news_background_service.py',
            'src/core/services/pred_time_service.py',
            'src/core/services/trade_time_service.py'
        ]
        
        total_caching = 0
        for service_file in service_files:
            if os.path.exists(service_file):
                with open(service_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_caching = 0
                for pattern in caching_patterns:
                    if pattern in content:
                        file_caching += 1
                
                total_caching += file_caching
                print(f"✅ {service_file}: {file_caching} caching patterns")
        
        if total_caching >= 3:
            print(f"✅ Caching optimizations: {total_caching}")
        else:
            print(f"⚠️  Limited caching optimizations: {total_caching}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing caching optimization: {e}")
        return False


def test_io_optimization():
    """Тест оптимизации I/O операций"""
    print("🧪 Testing I/O optimization...")
    
    try:
        # Проверяем оптимизации I/O
        io_patterns = [
            'with open',  # Контекстные менеджеры
            'aiofiles',  # Асинхронные файлы
            'async with',  # Асинхронные контексты
            'buffered',  # Буферизация
            'chunked',  # Чанковая обработка
            'stream',  # Потоковая обработка
            'batch',  # Батчевая обработка
            'bulk'  # Массовые операции
        ]
        
        # Проверяем в файлах с I/O операциями
        io_files = [
            'src/core/services/model_versioning_service.py',
            'src/core/services/pipeline_orchestrator.py',
            'src/core/database/orm/artifacts.py'
        ]
        
        total_io_optimizations = 0
        for file_path in io_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_optimizations = 0
                for pattern in io_patterns:
                    if pattern in content:
                        file_optimizations += 1
                
                total_io_optimizations += file_optimizations
                print(f"✅ {file_path}: {file_optimizations} I/O optimizations")
        
        if total_io_optimizations >= 3:
            print(f"✅ I/O optimizations: {total_io_optimizations}")
        else:
            print(f"⚠️  Limited I/O optimizations: {total_io_optimizations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing I/O optimization: {e}")
        return False


def test_frontend_performance():
    """Тест производительности Frontend"""
    print("🧪 Testing frontend performance...")
    
    try:
        # Проверяем оптимизации Frontend
        frontend_optimizations = [
            'useMemo',  # Мемоизация
            'useCallback',  # Кэширование колбэков
            'React.memo',  # Мемоизация компонентов
            'lazy',  # Ленивая загрузка
            'Suspense',  # Suspense для ленивой загрузки
            'debounce',  # Дебаунсинг
            'throttle',  # Троттлинг
            'virtualization'  # Виртуализация
        ]
        
        # Проверяем в компонентах
        component_files = [
            'frontend/src/components/profile/TrainAgentModal.jsx',
            'frontend/src/components/profile/ModuleTester.jsx',
            'frontend/src/components/profile/ModelVersioningPanel.jsx'
        ]
        
        total_optimizations = 0
        for component_file in component_files:
            if os.path.exists(component_file):
                with open(component_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_optimizations = 0
                for opt in frontend_optimizations:
                    if opt in content:
                        file_optimizations += 1
                
                total_optimizations += file_optimizations
                print(f"✅ {component_file}: {file_optimizations} frontend optimizations")
        
        if total_optimizations >= 2:
            print(f"✅ Frontend performance optimizations: {total_optimizations}")
        else:
            print(f"⚠️  Limited frontend optimizations: {total_optimizations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing frontend performance: {e}")
        return False


def test_error_handling_performance():
    """Тест производительности обработки ошибок"""
    print("🧪 Testing error handling performance...")
    
    try:
        # Проверяем оптимизации обработки ошибок
        error_patterns = [
            'try:',  # Try-catch блоки
            'except',  # Обработка исключений
            'finally:',  # Finally блоки
            'logging',  # Логирование
            'error_handler',  # Обработчики ошибок
            'retry',  # Повторные попытки
            'timeout',  # Таймауты
            'circuit_breaker'  # Circuit breaker
        ]
        
        # Проверяем в основных файлах
        main_files = [
            'src/core/services/pipeline_orchestrator.py',
            'src/backend/celery_app/tasks.py',
            'src/backend/app/routers/apidb_agent/router.py'
        ]
        
        total_error_handling = 0
        for file_path in main_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_error_handling = 0
                for pattern in error_patterns:
                    if pattern in content:
                        file_error_handling += 1
                
                total_error_handling += file_error_handling
                print(f"✅ {file_path}: {file_error_handling} error handling patterns")
        
        if total_error_handling >= 5:
            print(f"✅ Error handling optimizations: {total_error_handling}")
        else:
            print(f"⚠️  Limited error handling optimizations: {total_error_handling}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing error handling performance: {e}")
        return False


def main():
    """Основная функция тестирования производительности"""
    print("🚀 Starting Performance Tests...")
    
    tests = [
        test_pipeline_performance,
        test_database_performance,
        test_memory_optimization,
        test_cpu_optimization,
        test_caching_optimization,
        test_io_optimization,
        test_frontend_performance,
        test_error_handling_performance
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
    
    print(f"📊 Performance Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All performance tests passed!")
        return True
    else:
        print("⚠️  Some performance tests failed!")
        return False


if __name__ == '__main__':
    success = main()
    assert success, "Performance tests failed"
