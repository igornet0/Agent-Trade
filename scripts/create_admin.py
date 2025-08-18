#!/usr/bin/env python3
"""
Скрипт для создания администратора в базе данных
Использование: python scripts/create_admin.py [--login LOGIN] [--email EMAIL] [--password PASSWORD]
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.database import db_helper
from core.database.orm.users import orm_add_user
from core.utils import setup_logging
import hashlib


def hash_password(password: str) -> str:
    """Хеширует пароль с использованием SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


async def create_admin_user(login: str, email: str, password: str) -> bool:
    """Создает администратора в базе данных"""
    try:
        setup_logging()
        
        # Хешируем пароль
        hashed_password = hash_password(password)
        
        async with db_helper.get_session() as session:
            # Создаем пользователя с ролью admin
            user = await orm_add_user(
                session=session,
                login=login,
                hashed_password=hashed_password,
                email=email
            )
            
            # Обновляем роль на admin
            user.role = "admin"
            await session.commit()
            
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
