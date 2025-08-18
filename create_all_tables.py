#!/usr/bin/env python3
"""
Скрипт для создания всех необходимых таблиц в базе данных
"""

import asyncio
import asyncpg
import sys

async def create_all_tables():
    """Создает все необходимые таблицы в базе данных"""
    try:
        # Подключаемся к базе данных
        conn = await asyncpg.connect(
            user='agent',
            password='agent',
            database='agent',
            host='localhost',
            port=5432
        )
        
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
        
        await conn.close()
        print("✅ Все таблицы успешно созданы!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании таблиц: {e}")
        return False

async def main():
    print("🔧 Создание всех необходимых таблиц...")
    success = await create_all_tables()
    
    if success:
        print("\n🎉 Все таблицы готовы!")
    else:
        print("\n💥 Не удалось создать таблицы.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
