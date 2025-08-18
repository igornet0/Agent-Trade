#!/usr/bin/env python3
"""
Скрипт для создания всех необходимых таблиц внутри Docker контейнера
"""

import asyncio
import hashlib
import asyncpg
import sys

async def create_all_tables_docker_container():
    """Создает все необходимые таблицы в Docker PostgreSQL"""
    try:
        # Подключаемся к Docker PostgreSQL (внутри контейнера)
        conn = await asyncpg.connect(
            user='agent',
            password='agent',
            database='agent',
            host='postgres',  # Имя сервиса в docker-compose
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
        
        # Создаем таблицу coins
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS coins (
                id SERIAL PRIMARY KEY,
                name VARCHAR(50) UNIQUE NOT NULL,
                price_now FLOAT DEFAULT 0,
                max_price_now FLOAT DEFAULT 0,
                min_price_now FLOAT DEFAULT 0,
                open_price_now FLOAT DEFAULT 0,
                volume_now FLOAT DEFAULT 0,
                price_change_percentage_24h FLOAT,
                news_score_global FLOAT DEFAULT 100,
                parsed BOOLEAN DEFAULT TRUE
            )
        """)
        print("✅ Таблица coins создана!")
        
        # Создаем таблицу timeseries
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS timeseries (
                id SERIAL PRIMARY KEY,
                coin_id INTEGER REFERENCES coins(id),
                timestamp VARCHAR(50),
                path_dataset VARCHAR(100) UNIQUE
            )
        """)
        print("✅ Таблица timeseries создана!")
        
        # Создаем таблицу data_timeseries
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS data_timeseries (
                id SERIAL PRIMARY KEY,
                timeseries_id INTEGER REFERENCES timeseries(id),
                datetime TIMESTAMP NOT NULL,
                open FLOAT,
                max FLOAT,
                min FLOAT,
                close FLOAT,
                volume FLOAT
            )
        """)
        print("✅ Таблица data_timeseries создана!")
        
        # Создаем таблицу portfolio
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                coin_id INTEGER REFERENCES coins(id),
                amount FLOAT DEFAULT 0.0,
                price_avg FLOAT DEFAULT 0.0,
                CONSTRAINT uq_portfolio_user_coin UNIQUE (user_id, coin_id),
                CONSTRAINT ck_portfolio_amount_non_negative CHECK (amount >= 0)
            )
        """)
        print("✅ Таблица portfolio создана!")
        
        # Создаем таблицу transactions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                status VARCHAR(30) DEFAULT 'open',
                user_id INTEGER REFERENCES users(id),
                coin_id INTEGER REFERENCES coins(id),
                type VARCHAR(20) NOT NULL,
                amount_orig FLOAT NOT NULL,
                amount FLOAT NOT NULL,
                price FLOAT NOT NULL
            )
        """)
        print("✅ Таблица transactions создана!")
        
        # Создаем администратора
        hashed_password = hashlib.sha256('admin123'.encode()).hexdigest()
        
        # Проверяем, существует ли пользователь
        user = await conn.fetchrow(
            "SELECT id FROM users WHERE login = $1 OR email = $2",
            'admin', 'admin@agent-trade.com'
        )
        
        if not user:
            # Создаем пользователя
            await conn.execute("""
                INSERT INTO users (login, email, password, role, active, balance)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, 'admin', 'admin@agent-trade.com', hashed_password, 'admin', True, 0.0)
            print("✅ Администратор создан!")
        else:
            print("✅ Администратор уже существует!")
        
        await conn.close()
        print("✅ Все таблицы и администратор успешно созданы!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании таблиц: {e}")
        return False

async def main():
    print("🔧 Создание всех необходимых таблиц в Docker PostgreSQL...")
    success = await create_all_tables_docker_container()
    
    if success:
        print("\n🎉 Все таблицы и администратор готовы!")
        print("   Логин: admin")
        print("   Email: admin@agent-trade.com")
        print("   Пароль: admin123")
        print("   Роль: admin")
    else:
        print("\n💥 Не удалось создать таблицы.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
