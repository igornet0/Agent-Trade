# Система управления множественными торговыми агентами
# Создает ансамбль агентов для разных монет и стратегий

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from crypto_trading_agent import AdaptiveLearningAgent, train_crypto_trading_agent, test_agent_trading

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAgentEnsemble:
    """Ансамбль множественных торговых агентов"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.agents = {}
        self.agent_weights = {}
        self.ensemble_performance = []
        self.global_portfolio = {
            'total_value': 10000,  # Начальный капитал
            'positions': {},        # Позиции по монетам
            'cash': 10000,         # Наличные
            'unrealized_pnl': 0,   # Нереализованная прибыль/убыток
            'realized_pnl': 0      # Реализованная прибыль/убыток
        }
        
    def add_agent(self, coin_name: str, agent: AdaptiveLearningAgent, weight: float = 1.0):
        """Добавляет агента в ансамбль"""
        self.agents[coin_name] = agent
        self.agent_weights[coin_name] = weight
        
        # Нормализуем веса
        self._normalize_weights()
        
        logger.info(f"Добавлен агент для {coin_name} с весом {weight:.3f}")
    
    def _normalize_weights(self):
        """Нормализует веса агентов"""
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            for coin in self.agent_weights:
                self.agent_weights[coin] /= total_weight
    
    def train_all_agents(self, coin_list: List[str], epochs: int = 10):
        """Обучает всех агентов в ансамбле"""
        logger.info(f"Начинаем обучение {len(coin_list)} агентов")
        
        for coin_name in coin_list:
            try:
                logger.info(f"Обучение агента для {coin_name}")
                
                # Создаем и обучаем агента
                agent = train_crypto_trading_agent(coin_name)
                
                if agent:
                    # Добавляем в ансамбль с равным весом
                    self.add_agent(coin_name, agent, weight=1.0)
                    
                    # Тестируем агента
                    test_results = test_agent_trading(agent, test_period_days=3)
                    
                    # Обновляем вес на основе производительности
                    performance = agent.get_performance_summary()
                    if 'success_rate' in performance:
                        success_rate = performance['success_rate']
                        # Корректируем вес на основе успешности
                        new_weight = max(0.1, min(2.0, success_rate * 2))
                        self.agent_weights[coin_name] = new_weight
                        
                        logger.info(f"Агент {coin_name}: успешность {success_rate:.3f}, вес {new_weight:.3f}")
                    
                else:
                    logger.warning(f"Не удалось создать агента для {coin_name}")
                    
            except Exception as e:
                logger.error(f"Ошибка при обучении агента {coin_name}: {e}")
                continue
        
        # Нормализуем веса после обучения
        self._normalize_weights()
        logger.info("Обучение всех агентов завершено")
    
    def make_ensemble_decision(self, coin_name: str, current_data: pd.DataFrame) -> Dict:
        """Принимает торговое решение используя ансамбль агентов"""
        if coin_name not in self.agents:
            return {'error': f'Агент для {coin_name} не найден'}
        
        agent = self.agents[coin_name]
        weight = self.agent_weights[coin_name]
        
        # Получаем решение от агента
        decision = agent.make_trading_decision(current_data)
        
        # Добавляем информацию о весе агента
        decision['agent_weight'] = weight
        decision['coin_name'] = coin_name
        
        return decision
    
    def execute_ensemble_trades(self, decisions: List[Dict]) -> List[Dict]:
        """Выполняет торговые операции для всех агентов"""
        trade_results = []
        
        for decision in decisions:
            if 'error' in decision:
                continue
                
            coin_name = decision['coin_name']
            agent = self.agents[coin_name]
            
            # Выполняем сделку
            trade_result = agent.execute_trade(decision)
            trade_result['coin_name'] = coin_name
            trade_result['agent_weight'] = decision['agent_weight']
            
            trade_results.append(trade_result)
            
            # Обновляем портфель
            self._update_portfolio(trade_result)
        
        return trade_results
    
    def _update_portfolio(self, trade_result: Dict):
        """Обновляет глобальный портфель"""
        coin_name = trade_result['coin_name']
        action = trade_result['action']
        price = trade_result['price']
        pnl = trade_result['profit_loss']
        
        if action == 'BUY':
            if coin_name not in self.global_portfolio['positions']:
                self.global_portfolio['positions'][coin_name] = {
                    'quantity': 1,
                    'entry_price': price,
                    'current_price': price
                }
                self.global_portfolio['cash'] -= price
            else:
                # Увеличиваем позицию
                position = self.global_portfolio['positions'][coin_name]
                position['quantity'] += 1
                # Пересчитываем среднюю цену входа
                total_cost = position['entry_price'] * (position['quantity'] - 1) + price
                position['entry_price'] = total_cost / position['quantity']
                self.global_portfolio['cash'] -= price
                
        elif action == 'SELL':
            if coin_name in self.global_portfolio['positions']:
                position = self.global_portfolio['positions'][coin_name]
                if position['quantity'] > 0:
                    position['quantity'] -= 1
                    self.global_portfolio['cash'] += price
                    
                    # Если позиция закрыта полностью
                    if position['quantity'] == 0:
                        del self.global_portfolio['positions'][coin_name]
                    
                    # Обновляем P&L
                    if pnl != 0:
                        self.global_portfolio['realized_pnl'] += pnl
        
        # Обновляем нереализованную прибыль
        self._update_unrealized_pnl()
    
    def _update_unrealized_pnl(self):
        """Обновляет нереализованную прибыль/убыток"""
        total_unrealized = 0
        
        for coin_name, position in self.global_portfolio['positions'].items():
            # Здесь нужно получить текущую цену монеты
            # Для простоты используем последнюю известную цену
            current_price = position['current_price']
            entry_price = position['entry_price']
            quantity = position['quantity']
            
            unrealized_pnl = (current_price - entry_price) * quantity
            total_unrealized += unrealized_pnl
        
        self.global_portfolio['unrealized_pnl'] = total_unrealized
    
    def get_portfolio_summary(self) -> Dict:
        """Возвращает сводку по портфелю"""
        total_value = self.global_portfolio['cash']
        
        for coin_name, position in self.global_portfolio['positions'].items():
            total_value += position['current_price'] * position['quantity']
        
        return {
            'total_value': total_value,
            'cash': self.global_portfolio['cash'],
            'positions': self.global_portfolio['positions'],
            'unrealized_pnl': self.global_portfolio['unrealized_pnl'],
            'realized_pnl': self.global_portfolio['realized_pnl'],
            'total_return': (total_value - 10000) / 10000 * 100,  # В процентах
            'agent_weights': self.agent_weights
        }
    
    def adaptive_weight_adjustment(self):
        """Адаптивно корректирует веса агентов на основе производительности"""
        logger.info("Начинаем адаптивную корректировку весов")
        
        performance_scores = {}
        
        for coin_name, agent in self.agents.items():
            # Получаем производительность агента
            performance = agent.get_performance_summary()
            
            if 'success_rate' in performance and 'total_pnl' in performance:
                # Комбинированная оценка: успешность + P&L
                success_rate = performance['success_rate']
                total_pnl = performance['total_pnl']
                
                # Нормализуем P&L (предполагаем, что максимальный P&L может быть 1000)
                normalized_pnl = max(-1, min(1, total_pnl / 1000))
                
                # Комбинированная оценка
                combined_score = 0.7 * success_rate + 0.3 * normalized_pnl
                performance_scores[coin_name] = max(0.1, combined_score)
                
                logger.info(f"Агент {coin_name}: успешность {success_rate:.3f}, P&L {total_pnl:.4f}, "
                          f"общая оценка {combined_score:.3f}")
        
        # Обновляем веса на основе производительности
        if performance_scores:
            total_score = sum(performance_scores.values())
            if total_score > 0:
                for coin_name in self.agent_weights:
                    if coin_name in performance_scores:
                        # Плавное обновление весов
                        current_weight = self.agent_weights[coin_name]
                        target_weight = performance_scores[coin_name] / total_score
                        
                        # Обновляем с коэффициентом 0.3
                        new_weight = 0.7 * current_weight + 0.3 * target_weight
                        self.agent_weights[coin_name] = new_weight
                        
                        logger.info(f"Обновлен вес {coin_name}: {current_weight:.3f} -> {new_weight:.3f}")
        
        # Нормализуем веса
        self._normalize_weights()
        
        # Сохраняем метрики
        self.ensemble_performance.append({
            'timestamp': datetime.now(),
            'agent_weights': self.agent_weights.copy(),
            'portfolio_summary': self.get_portfolio_summary()
        })
        
        logger.info("Адаптивная корректировка весов завершена")
    
    def save_ensemble_state(self, filepath: str):
        """Сохраняет состояние ансамбля"""
        state = {
            'agent_weights': self.agent_weights,
            'global_portfolio': self.global_portfolio,
            'ensemble_performance': self.ensemble_performance,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Состояние ансамбля сохранено в {filepath}")
    
    def load_ensemble_state(self, filepath: str):
        """Загружает состояние ансамбля"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.agent_weights = state['agent_weights']
            self.global_portfolio = state['global_portfolio']
            self.ensemble_performance = state['ensemble_performance']
            
            logger.info(f"Состояние ансамбля загружено из {filepath}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке состояния: {e}")

class RealTimeTradingSystem:
    """Система торговли в реальном времени"""
    
    def __init__(self, ensemble: MultiAgentEnsemble):
        self.ensemble = ensemble
        self.is_running = False
        self.trading_interval = 300  # 5 минут в секундах
        self.last_trade_time = None
        
    def start_trading(self):
        """Запускает торговлю в реальном времени"""
        self.is_running = True
        logger.info("Система торговли в реальном времени запущена")
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Проверяем, прошло ли достаточно времени с последней сделки
                if (self.last_trade_time is None or 
                    (current_time - self.last_trade_time).seconds >= self.trading_interval):
                    
                    # Выполняем торговый цикл
                    self._trading_cycle()
                    self.last_trade_time = current_time
                
                # Небольшая пауза
                import time
                time.sleep(60)  # Проверяем каждую минуту
                
            except KeyboardInterrupt:
                logger.info("Получен сигнал остановки")
                self.stop_trading()
                break
            except Exception as e:
                logger.error(f"Ошибка в торговом цикле: {e}")
                continue
    
    def stop_trading(self):
        """Останавливает торговлю"""
        self.is_running = False
        logger.info("Система торговли остановлена")
    
    def _trading_cycle(self):
        """Выполняет один торговый цикл"""
        logger.info("Выполняем торговый цикл")
        
        # Получаем решения от всех агентов
        decisions = []
        
        for coin_name, agent in self.ensemble.agents.items():
            try:
                # Получаем последние данные (здесь нужно реализовать получение реальных данных)
                # Для демонстрации используем последние данные из агента
                current_data = agent.data.tail(agent.seq_len)
                
                decision = self.ensemble.make_ensemble_decision(coin_name, current_data)
                decisions.append(decision)
                
            except Exception as e:
                logger.error(f"Ошибка при получении решения для {coin_name}: {e}")
                continue
        
        # Выполняем сделки
        if decisions:
            trade_results = self.ensemble.execute_ensemble_trades(decisions)
            
            # Логируем результаты
            for result in trade_results:
                if result['action'] != 'HOLD':
                    logger.info(f"Сделка: {result['coin_name']} {result['action']} "
                              f"по цене {result['price']:.4f}, P&L: {result['profit_loss']:.4f}")
        
        # Показываем сводку по портфелю
        portfolio_summary = self.ensemble.get_portfolio_summary()
        logger.info(f"Портфель: общая стоимость {portfolio_summary['total_value']:.2f}, "
                   f"P&L: {portfolio_summary['realized_pnl']:.2f}")
        
        # Периодически корректируем веса
        if len(self.ensemble.ensemble_performance) % 10 == 0:  # Каждые 10 циклов
            self.ensemble.adaptive_weight_adjustment()

# Функция для создания и обучения ансамбля агентов
def create_and_train_ensemble(coin_list: List[str], device: torch.device) -> MultiAgentEnsemble:
    """Создает и обучает ансамбль агентов"""
    logger.info(f"Создание ансамбля для {len(coin_list)} монет")
    
    # Создаем ансамбль
    ensemble = MultiAgentEnsemble(device)
    
    # Обучаем всех агентов
    ensemble.train_all_agents(coin_list)
    
    # Показываем финальные веса
    logger.info("Финальные веса агентов:")
    for coin, weight in ensemble.agent_weights.items():
        logger.info(f"{coin}: {weight:.3f}")
    
    return ensemble

# Функция для запуска торговли в реальном времени
def run_real_time_trading(ensemble: MultiAgentEnsemble):
    """Запускает торговлю в реальном времени"""
    logger.info("Запуск системы торговли в реальном времени")
    
    # Создаем систему торговли
    trading_system = RealTimeTradingSystem(ensemble)
    
    try:
        # Запускаем торговлю
        trading_system.start_trading()
    except KeyboardInterrupt:
        logger.info("Торговля остановлена пользователем")
    finally:
        # Сохраняем состояние
        ensemble.save_ensemble_state(f"ensemble_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# Основная функция
if __name__ == "__main__":
    # Список монет для торговли
    coins_to_trade = ["BTC", "ETH", "ADA", "DOT", "LINK"]
    
    # Создаем и обучаем ансамбль
    ensemble = create_and_train_ensemble(coins_to_trade, device)
    
    # Запускаем торговлю в реальном времени
    run_real_time_trading(ensemble)
