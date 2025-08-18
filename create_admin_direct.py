#!/usr/bin/env python3
"""
Скрипт для создания таблицы users и администратора напрямую
"""

import asyncio
import hashlib
import asyncpg
import sys

async def create_admin_user():
    """Создает таблицу users и администратора напрямую в базе данных"""
    try:
        # Подключаемся к базе данных
        conn = await asyncpg.connect(
            user='agent',
            password='agent',
            database='agent',
            host='localhost',
            port=5432
        )
        
        # Создаем таблицу users
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                login VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(50) UNIQUE,
                password VARCHAR(255) NOT NULL,
                user_telegram_id BIGINT,
                balance FLOAT DEFAULT 0,
                role VARCHAR(50) DEFAULT 'user',
                active BOOLEAN DEFAULT TRUE
            )
        """)
        
        print("✅ Таблица users создана!")
        
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
    print("🔧 Создание таблицы users и администратора...")
    success = await create_admin_user()
    
    if success:
        print("\n🎉 Администратор готов к использованию!")
        print("   Вы можете войти в систему используя указанные учетные данные.")
    else:
        print("\n💥 Не удалось создать администратора.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
