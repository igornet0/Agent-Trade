#!/usr/bin/env python3
"""
Локальный скрипт для создания администратора в базе данных
Использование: python create_admin_local.py [--login LOGIN] [--email EMAIL] [--password PASSWORD]
"""

import asyncio
import argparse
import sys
import os
import asyncpg
import hashlib


def hash_password(password: str) -> str:
    """Хеширует пароль с использованием SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


async def create_admin_user(login: str, email: str, password: str) -> bool:
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
        hashed_password = hash_password(password)
        
        # Проверяем, существует ли пользователь
        user = await conn.fetchrow(
            "SELECT id FROM users WHERE login = $1 OR email = $2",
            login, email
        )
        
        if user:
            print("✅ Пользователь admin уже существует!")
            await conn.close()
            return True
        
        # Создаем таблицу users, если она не существует
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
        
        # Создаем администратора
        await conn.execute("""
            INSERT INTO users (login, email, password, role, active, balance)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, login, email, hashed_password, 'admin', True, 0.0)
        
        await conn.close()
        
        print(f"✅ Администратор успешно создан!")
        print(f"   Логин: {login}")
        print(f"   Email: {email}")
        print(f"   Роль: admin")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании администратора: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Создание администратора в базе данных')
    parser.add_argument('--login', default='admin', help='Логин администратора (по умолчанию: admin)')
    parser.add_argument('--email', default='admin@agent-trade.com', help='Email администратора (по умолчанию: admin@agent-trade.com)')
    parser.add_argument('--password', default='admin123', help='Пароль администратора (по умолчанию: admin123)')
    
    args = parser.parse_args()
    
    print(f"🔧 Создание администратора...")
    print(f"   Логин: {args.login}")
    print(f"   Email: {args.email}")
    print(f"   Пароль: {args.password}")
    print()
    
    # Запускаем асинхронную функцию
    success = asyncio.run(create_admin_user(args.login, args.email, args.password))
    
    if success:
        print("\n🎉 Администратор готов к использованию!")
        print("   Вы можете войти в систему используя указанные учетные данные.")
    else:
        print("\n💥 Не удалось создать администратора.")
        sys.exit(1)


if __name__ == "__main__":
    main()
