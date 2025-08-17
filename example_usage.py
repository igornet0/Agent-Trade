# Примеры использования системы обучения торговых агентов
# Демонстрирует различные сценарии обучения и торговли

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импортируем наши модули
from crypto_trading_agent import (
    AdaptiveLearningAgent, 
    train_crypto_trading_agent, 
    test_agent_trading,
    TradingLabelGenerator,
    CryptoDataset
)

from multi_agent_ensemble import (
    MultiAgentEnsemble,
    create_and_train_ensemble,
    run_real_time_trading
)

# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

def example_1_single_agent_training():
    """Пример 1: Обучение одного агента для BTC"""
    print("\n" + "="*50)
    print("ПРИМЕР 1: Обучение одного агента для BTC")
    print("="*50)
    
    try:
        # Обучаем агента для BTC
        btc_agent = train_crypto_trading_agent("BTC")
        
        if btc_agent:
            print("✅ Агент успешно обучен!")
            
            # Тестируем агента
            print("\nТестируем агента на исторических данных...")
            test_results = test_agent_trading(btc_agent, test_period_days=3)
            
            # Показываем производительность
            performance = btc_agent.get_performance_summary()
            print("\n📊 Производительность агента:")
            for key, value in performance.items():
                if key != 'model_weights':
                    print(f"  {key}: {value}")
            
            return btc_agent
        else:
            print("❌ Не удалось обучить агента")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        return None

def example_2_ensemble_training():
    """Пример 2: Создание и обучение ансамбля агентов"""
    print("\n" + "="*50)
    print("ПРИМЕР 2: Создание и обучение ансамбля агентов")
    print("="*50)
    
    try:
        # Список монет для торговли
        coins_to_trade = ["BTC", "ETH", "ADA"]
        
        print(f"Создаем ансамбль для {len(coins_to_trade)} монет: {', '.join(coins_to_trade)}")
        
        # Создаем и обучаем ансамбль
        ensemble = create_and_train_ensemble(coins_to_trade, device)
        
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
        
        return ensemble
        
    except Exception as e:
        print(f"❌ Ошибка при создании ансамбля: {e}")
        return None

def example_3_custom_agent_configuration():
    """Пример 3: Настройка агента с пользовательскими параметрами"""
    print("\n" + "="*50)
    print("ПРИМЕР 3: Настройка агента с пользовательскими параметрами")
    print("="*50)
    
    try:
        # Создаем пользовательский генератор меток
        custom_label_generator = TradingLabelGenerator(
            profit_threshold=0.015,  # 1.5% прибыли
            stop_loss=0.008          # 0.8% убытка
        )
        
        print("Создан пользовательский генератор меток:")
        print(f"  Порог прибыли: {custom_label_generator.profit_threshold:.3f}")
        print(f"  Стоп-лосс: {custom_label_generator.stop_loss:.3f}")
        
        # Создаем пользовательский датасет
        from core import data_helper
        from backend.Dataset import DatasetTimeseries
        
        # Получаем данные для ETH
        eth_file = None
        for file in data_helper.get_path("processed", timetravel="5m"):
            if "ETH" in str(file):
                eth_file = file
                break
        
        if eth_file:
            eth_data = DatasetTimeseries(eth_file).get_dataset()
            print(f"\nЗагружены данные ETH: {eth_data.shape}")
            
            # Создаем пользовательский датасет
            custom_dataset = CryptoDataset(
                data=eth_data,
                seq_len=60,      # Увеличенная длина последовательности
                pred_len=8,      # Увеличенное количество предсказаний
                normalize=True
            )
            
            print(f"Создан пользовательский датасет:")
            print(f"  Размер: {len(custom_dataset)}")
            print(f"  Длина последовательности: {custom_dataset.seq_len}")
            print(f"  Количество предсказаний: {custom_dataset.pred_len}")
            
            # Создаем агента с пользовательскими параметрами
            custom_agent = AdaptiveLearningAgent(
                coin_name="ETH_CUSTOM",
                data=eth_data,
                device=device
            )
            
            # Настраиваем параметры
            custom_agent.seq_len = 60
            custom_agent.pred_len = 8
            custom_agent.learning_rate = 0.0005  # Меньший learning rate
            custom_agent.batch_size = 64         # Больший batch size
            custom_agent.epochs = 15             # Больше эпох
            
            print(f"\nСоздан пользовательский агент:")
            print(f"  Learning rate: {custom_agent.learning_rate}")
            print(f"  Batch size: {custom_agent.batch_size}")
            print(f"  Epochs: {custom_agent.epochs}")
            
            return custom_agent
            
        else:
            print("❌ Не удалось найти данные для ETH")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка при настройке агента: {e}")
        return None

def example_4_backtesting_strategy():
    """Пример 4: Бэктестинг торговой стратегии"""
    print("\n" + "="*50)
    print("ПРИМЕР 4: Бэктестинг торговой стратегии")
    print("="*50)
    
    try:
        # Создаем простого агента для бэктестинга
        from core import data_helper
        from backend.Dataset import DatasetTimeseries
        
        # Получаем данные для DOT
        dot_file = None
        for file in data_helper.get_path("processed", timetravel="5m"):
            if "DOT" in str(file):
                dot_file = file
                break
        
        if dot_file:
            dot_data = DatasetTimeseries(dot_file).get_dataset()
            print(f"Загружены данные DOT: {dot_data.shape}")
            
            # Создаем агента
            backtest_agent = AdaptiveLearningAgent(
                coin_name="DOT_BACKTEST",
                data=dot_data,
                device=device
            )
            
            # Обучаем на исторических данных
            print("Обучаем агента на исторических данных...")
            train_loader, val_loader = backtest_agent.prepare_data()
            backtest_agent.train_models(train_loader, val_loader)
            
            # Бэктестируем на последних данных
            print("\nЗапускаем бэктестинг...")
            backtest_results = test_agent_trading(backtest_agent, test_period_days=7)
            
            # Анализируем результаты
            performance = backtest_agent.get_performance_summary()
            
            print("\n📊 Результаты бэктестинга:")
            print(f"  Всего сделок: {performance['total_trades']}")
            print(f"  Успешных сделок: {performance['successful_trades']}")
            print(f"  Процент успеха: {performance['success_rate']:.2%}")
            print(f"  Общий P&L: ${performance['total_pnl']:.4f}")
            print(f"  Средний P&L: ${performance['average_pnl']:.4f}")
            
            # Анализируем детали сделок
            if backtest_results:
                print(f"\n📈 Детали сделок:")
                for i, result in enumerate(backtest_results[:5]):  # Показываем первые 5
                    print(f"  Сделка {i+1}: {result['action']} по цене ${result['price']:.4f}, "
                          f"P&L: ${result['profit_loss']:.4f}")
                
                if len(backtest_results) > 5:
                    print(f"  ... и еще {len(backtest_results) - 5} сделок")
            
            return backtest_agent
            
        else:
            print("❌ Не удалось найти данные для DOT")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка при бэктестинге: {e}")
        return None

def example_5_risk_management():
    """Пример 5: Управление рисками и стоп-лоссы"""
    print("\n" + "="*50)
    print("ПРИМЕР 5: Управление рисками и стоп-лоссы")
    print("="*50)
    
    try:
        # Создаем агента с настройками управления рисками
        from core import data_helper
        from backend.Dataset import DatasetTimeseries
        
        # Получаем данные для LINK
        link_file = None
        for file in data_helper.get_path("processed", timetravel="5m"):
            if "LINK" in str(file):
                link_file = file
                break
        
        if link_file:
            link_data = DatasetTimeseries(link_file).get_dataset()
            print(f"Загружены данные LINK: {link_data.shape}")
            
            # Создаем агента с настройками риска
            risk_agent = AdaptiveLearningAgent(
                coin_name="LINK_RISK",
                data=link_data,
                device=device
            )
            
            # Настраиваем параметры управления рисками
            risk_agent.stop_loss = 0.015        # 1.5% стоп-лосс
            risk_agent.take_profit = 0.03       # 3% тейк-профит
            
            print("Настроены параметры управления рисками:")
            print(f"  Стоп-лосс: {risk_agent.stop_loss:.3f}")
            print(f"  Тейк-профит: {risk_agent.take_profit:.3f}")
            
            # Обучаем агента
            print("\nОбучаем агента...")
            train_loader, val_loader = risk_agent.prepare_data()
            risk_agent.train_models(train_loader, val_loader)
            
            # Тестируем с управлением рисками
            print("\nТестируем стратегию управления рисками...")
            test_results = test_agent_trading(risk_agent, test_period_days=5)
            
            # Анализируем результаты
            performance = risk_agent.get_performance_summary()
            
            print("\n📊 Результаты с управлением рисками:")
            print(f"  Всего сделок: {performance['total_trades']}")
            print(f"  Процент успеха: {performance['success_rate']:.2%}")
            print(f"  Общий P&L: ${performance['total_pnl']:.4f}")
            
            # Анализируем распределение P&L
            if test_results:
                pnl_values = [result['profit_loss'] for result in test_results]
                max_profit = max(pnl_values) if pnl_values else 0
                max_loss = min(pnl_values) if pnl_values else 0
                
                print(f"\n📈 Анализ рисков:")
                print(f"  Максимальная прибыль: ${max_profit:.4f}")
                print(f"  Максимальный убыток: ${max_loss:.4f}")
                print(f"  Диапазон P&L: ${max_profit - max_loss:.4f}")
                
                # Вычисляем коэффициент Шарпа (упрощенно)
                if len(pnl_values) > 1:
                    mean_pnl = np.mean(pnl_values)
                    std_pnl = np.std(pnl_values)
                    if std_pnl > 0:
                        sharpe_ratio = mean_pnl / std_pnl
                        print(f"  Коэффициент Шарпа: {sharpe_ratio:.3f}")
            
            return risk_agent
            
        else:
            print("❌ Не удалось найти данные для LINK")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка при настройке управления рисками: {e}")
        return None

def example_6_ensemble_optimization():
    """Пример 6: Оптимизация ансамбля агентов"""
    print("\n" + "="*50)
    print("ПРИМЕР 6: Оптимизация ансамбля агентов")
    print("="*50)
    
    try:
        # Создаем ансамбль с несколькими монетами
        coins_to_optimize = ["BTC", "ETH", "ADA", "DOT", "LINK"]
        print(f"Создаем ансамбль для оптимизации: {', '.join(coins_to_optimize)}")
        
        # Создаем и обучаем ансамбль
        ensemble = create_and_train_ensemble(coins_to_optimize, device)
        
        if ensemble:
            print("✅ Ансамбль создан!")
            
            # Показываем начальные веса
            print("\n📊 Начальные веса агентов:")
            for coin, weight in ensemble.agent_weights.items():
                print(f"  {coin}: {weight:.3f}")
            
            # Выполняем несколько циклов адаптивной корректировки
            print("\n🔄 Выполняем адаптивную корректировку весов...")
            
            for cycle in range(3):
                print(f"\nЦикл корректировки {cycle + 1}/3:")
                
                # Симулируем торговлю для обновления производительности
                for coin_name, agent in ensemble.agents.items():
                    # Берем последние данные для тестирования
                    test_data = agent.data.tail(agent.seq_len * 2)
                    
                    # Принимаем несколько торговых решений
                    for i in range(5):
                        current_slice = test_data.iloc[i:i+agent.seq_len]
                        decision = agent.make_trading_decision(current_slice)
                        trade_result = agent.execute_trade(decision)
                
                # Корректируем веса
                ensemble.adaptive_weight_adjustment()
                
                # Показываем текущие веса
                print(f"  Веса после цикла {cycle + 1}:")
                for coin, weight in ensemble.agent_weights.items():
                    print(f"    {coin}: {weight:.3f}")
            
            # Показываем финальные веса
            print("\n📊 Финальные веса после оптимизации:")
            for coin, weight in ensemble.agent_weights.items():
                print(f"  {coin}: {weight:.3f}")
            
            # Показываем сводку по портфелю
            portfolio = ensemble.get_portfolio_summary()
            print(f"\n💰 Финальная сводка по портфелю:")
            print(f"  Общая стоимость: ${portfolio['total_value']:.2f}")
            print(f"  Общая доходность: {portfolio['total_return']:.2f}%")
            
            return ensemble
            
        else:
            print("❌ Не удалось создать ансамбль")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка при оптимизации ансамбля: {e}")
        return None

def main():
    """Основная функция с запуском всех примеров"""
    print("🚀 ЗАПУСК ПРИМЕРОВ ИСПОЛЬЗОВАНИЯ СИСТЕМЫ ОБУЧЕНИЯ ТОРГОВЫХ АГЕНТОВ")
    print("="*80)
    
    # Список примеров для выполнения
    examples = [
        ("Обучение одного агента для BTC", example_1_single_agent_training),
        ("Создание и обучение ансамбля агентов", example_2_ensemble_training),
        ("Настройка агента с пользовательскими параметрами", example_3_custom_agent_configuration),
        ("Бэктестинг торговой стратегии", example_4_backtesting_strategy),
        ("Управление рисками и стоп-лоссы", example_5_risk_management),
        ("Оптимизация ансамбля агентов", example_6_ensemble_optimization)
    ]
    
    results = {}
    
    # Выполняем примеры
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*20} ПРИМЕР {i}/6 {'='*20}")
        print(f"📋 {name}")
        
        try:
            result = func()
            results[name] = result
            print(f"✅ Пример {i} выполнен успешно")
        except Exception as e:
            print(f"❌ Ошибка в примере {i}: {e}")
            results[name] = None
    
    # Итоговая сводка
    print("\n" + "="*80)
    print("📊 ИТОГОВАЯ СВОДКА ВЫПОЛНЕНИЯ ПРИМЕРОВ")
    print("="*80)
    
    successful_examples = sum(1 for result in results.values() if result is not None)
    total_examples = len(examples)
    
    print(f"Всего примеров: {total_examples}")
    print(f"Успешно выполнено: {successful_examples}")
    print(f"Процент успеха: {successful_examples/total_examples*100:.1f}%")
    
    if successful_examples > 0:
        print("\n🎯 РЕКОМЕНДАЦИИ ПО ДАЛЬНЕЙШЕМУ ИСПОЛЬЗОВАНИЮ:")
        print("1. Начните с обучения одного агента для BTC (пример 1)")
        print("2. Создайте ансамбль для нескольких монет (пример 2)")
        print("3. Настройте параметры управления рисками (пример 5)")
        print("4. Оптимизируйте веса агентов (пример 6)")
        print("5. Запустите торговлю в реальном времени")
        
        print("\n💡 ДЛЯ ЗАПУСКА ТОРГОВЛИ В РЕАЛЬНОМ ВРЕМЕНИ:")
        print("python multi_agent_ensemble.py")
    
    print("\n" + "="*80)
    print("🏁 ВСЕ ПРИМЕРЫ ЗАВЕРШЕНЫ")

if __name__ == "__main__":
    main()
