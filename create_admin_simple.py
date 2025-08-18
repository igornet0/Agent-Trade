#!/usr/bin/env python3
"""
Простой скрипт для создания администратора напрямую в базе данных
"""

import asyncio
import hashlib
import asyncpg
import sys

async def create_admin_user():
    """Создает администратора напрямую в базе данных"""
    try:
        # Подключаемся к базе данных
        conn = await asyncpg.connect(
            user='agent',
            password='agent',
            database='agent',
            host='localhost',
            port=5432
        )
        
        # Хешируем пароль
        hashed_password = hashlib.sha256('admin123'.encode()).hexdigest()
        
        # Проверяем, существует ли пользователь
        user = await conn.fetchrow(
            "SELECT id FROM users WHERE login = $1 OR email = $2",
            'admin', 'admin@agent-trade.com'
        )
        
        if user:
            print("✅ Пользователь admin уже существует!")
            return True
        
        # Создаем пользователя
        await conn.execute("""
            INSERT INTO users (login, email, password, role, active, balance)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, 'admin', 'admin@agent-trade.com', hashed_password, 'admin', True, 0.0)
        
        await conn.close()
        
        print("✅ Администратор успешно создан!")
        print("   Логин: admin")
        print("   Email: admin@agent-trade.com")
        print("   Пароль: admin123")
        print("   Роль: admin")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании администратора: {e}")
        return False

async def main():
    print("🔧 Создание администратора...")
    success = await create_admin_user()
    
    if success:
        print("\n🎉 Администратор готов к использованию!")
        print("   Вы можете войти в систему используя указанные учетные данные.")
    else:
        print("\n💥 Не удалось создать администратора.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
