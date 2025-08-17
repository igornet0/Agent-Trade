#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска системы обучения торговых агентов
"""

import sys
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Основная функция запуска системы"""
    print("🚀 ЗАПУСК СИСТЕМЫ ОБУЧЕНИЯ ТОРГОВЫХ АГЕНТОВ")
    print("="*60)
    
    try:
        # Импортируем наши модули
        from crypto_trading_agent import train_crypto_trading_agent, test_agent_trading
        from multi_agent_ensemble import create_and_train_ensemble, run_real_time_trading
        
        import torch
        
        # Проверяем доступность GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {device}")
        
        # Список монет для торговли
        coins_to_trade = ["BTC", "ETH", "ADA", "DOT", "LINK"]
        print(f"\nСоздаем ансамбль для {len(coins_to_trade)} монет: {', '.join(coins_to_trade)}")
        
        # Создаем и обучаем ансамбль
        print("\n🔄 Начинаем обучение агентов...")
        ensemble = create_and_train_ensemble(coins_to_trade, device)
        
        if ensemble:
            print("✅ Ансамбль успешно создан и обучен!")
            
            # Показываем веса агентов
            print("\n📊 Веса агентов в ансамбле:")
            for coin, weight in ensemble.agent_weights.items():
                print(f"  {coin}: {weight:.3f}")
            
            # Показываем сводку по портфелю
            portfolio = ensemble.get_portfolio_summary()
            print("\n💰 Сводка по портфелю:")
            print(f"  Общая стоимость: ${portfolio['total_value']:.2f}")
            print(f"  Наличные: ${portfolio['cash']:.2f}")
            print(f"  Реализованный P&L: ${portfolio['realized_pnl']:.2f}")
            print(f"  Общая доходность: {portfolio['total_return']:.2f}%")
            
            # Спрашиваем пользователя о запуске торговли
            print("\n" + "="*60)
            print("🎯 ВЫБЕРИТЕ РЕЖИМ РАБОТЫ:")
            print("1. Демонстрационный режим (тестирование на исторических данных)")
            print("2. Торговля в реальном времени")
            print("3. Выход")
            
            while True:
                try:
                    choice = input("\nВведите номер (1-3): ").strip()
                    
                    if choice == "1":
                        print("\n🔄 Запускаем демонстрационный режим...")
                        run_demo_mode(ensemble)
                        break
                        
                    elif choice == "2":
                        print("\n🚨 ВНИМАНИЕ: Режим реальной торговли!")
                        print("Это может привести к реальным финансовым потерям!")
                        confirm = input("Вы уверены? (да/нет): ").strip().lower()
                        
                        if confirm in ['да', 'yes', 'y', 'д']:
                            print("\n🚀 Запускаем торговлю в реальном времени...")
                            run_real_time_trading(ensemble)
                        else:
                            print("Режим реальной торговли отменен.")
                            break
                            
                    elif choice == "3":
                        print("Выход из системы.")
                        break
                        
                    else:
                        print("Неверный выбор. Введите 1, 2 или 3.")
                        
                except KeyboardInterrupt:
                    print("\n\nПолучен сигнал остановки. Выход...")
                    break
                except Exception as e:
                    print(f"Ошибка: {e}")
                    continue
            
        else:
            print("❌ Не удалось создать ансамбль")
            return 1
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Убедитесь, что все модули установлены и доступны")
        return 1
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return 1
    
    print("\n🏁 Система завершена.")
    return 0

def run_demo_mode(ensemble):
    """Запускает демонстрационный режим"""
    print("\n📊 ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
    print("="*40)
    
    try:
        # Тестируем каждого агента на исторических данных
        for coin_name, agent in ensemble.agents.items():
            print(f"\n🔄 Тестируем агента {coin_name}...")
            
            # Тестируем на 3 днях данных
            test_results = test_agent_trading(agent, test_period_days=3)
            
            # Показываем результаты
            performance = agent.get_performance_summary()
            print(f"  Результаты для {coin_name}:")
            print(f"    Всего сделок: {performance['total_trades']}")
            print(f"    Процент успеха: {performance['success_rate']:.2%}")
            print(f"    Общий P&L: ${performance['total_pnl']:.4f}")
        
        # Показываем общую сводку
        print("\n📊 ОБЩАЯ СВОДКА:")
        portfolio = ensemble.get_portfolio_summary()
        print(f"  Общая стоимость портфеля: ${portfolio['total_value']:.2f}")
        print(f"  Общая доходность: {portfolio['total_return']:.2f}%")
        
        # Адаптивная корректировка весов
        print("\n🔄 Выполняем адаптивную корректировку весов...")
        ensemble.adaptive_weight_adjustment()
        
        print("\n✅ Демонстрационный режим завершен!")
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрационном режиме: {e}")
        logger.error(f"Ошибка в демонстрационном режиме: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nПолучен сигнал остановки. Выход...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
        sys.exit(1)
