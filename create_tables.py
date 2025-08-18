#!/usr/bin/env python3
"""
Скрипт для создания таблиц в базе данных
"""

import asyncio
import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.database import Base, db_helper
from core.utils import setup_logging

async def create_tables():
    """Создает таблицы в базе данных"""
    try:
        setup_logging()
        
        async with db_helper.get_session() as session:
            # Создаем таблицы
            await db_helper.init_db()
            print("✅ Таблицы успешно созданы!")
            return True
            
    except Exception as e:
        print(f"❌ Ошибка при создании таблиц: {e}")
        return False

async def main():
    print("🔧 Создание таблиц в базе данных...")
    success = await create_tables()
    
    if success:
        print("\n🎉 Таблицы готовы!")
    else:
        print("\n💥 Не удалось создать таблицы.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
