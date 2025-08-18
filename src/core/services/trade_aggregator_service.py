import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import math

try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError as e:
    logging.warning(f"ML libraries not available: {e}")
    xgb = None

from ..database.orm.market import orm_get_coin_data
from ..database.orm.news import orm_get_news_background
from ..utils.metrics import calculate_technical_indicators

logger = logging.getLogger(__name__)

class TradeAggregatorService:
    """Сервис для Trade (Aggregator) модуля - финальный исполнитель торговых решений"""

    def __init__(self):
        self.models_dir = "models/models_pth/AgentTrade"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Конфигурация по умолчанию
        self.default_config = {
            'mode': 'rules',  # 'rules' или 'ml'
            'weights': {
                'pred_time': 0.4,
                'trade_time': 0.4,
                'risk': 0.2
            },
            'thresholds': {
                'buy_threshold': 0.6,
                'sell_threshold': 0.4,
                'hold_threshold': 0.3
            },
            'risk_limits': {
                'max_position_size': 0.1,  # 10% от баланса
                'max_leverage': 3.0,
                'stop_loss_pct': 0.05,  # 5%
                'take_profit_pct': 0.15  # 15%
            },
            'portfolio': {
                'max_coins': 10,
                'rebalance_frequency': '1h',
                'correlation_threshold': 0.7
            }
        }

    def _calculate_portfolio_metrics(self, positions, balance):
        """Расчет метрик портфеля"""
        if not positions:
            return {
                'total_value': balance,
                'total_pnl': 0.0,
                'exposure': 0.0,
                'diversification': 1.0,
                'risk_score': 0.0
            }
        
        total_value = balance
        total_pnl = 0.0
        total_exposure = 0.0
        
        for pos in positions:
            total_value += pos.get('unrealized_pnl', 0)
            total_pnl += pos.get('unrealized_pnl', 0)
            total_exposure += abs(pos.get('size', 0) * pos.get('entry_price', 0))
        
        # Диверсификация (1.0 = максимальная)
        if len(positions) > 1:
            weights = [abs(pos.get('size', 0) * pos.get('entry_price', 0)) / total_exposure for pos in positions]
            diversification = 1.0 - sum(w**2 for w in weights)  # HHI index
        else:
            diversification = 0.0
        
        # Риск-скор (простая эвристика)
        risk_score = min(1.0, total_exposure / balance * 0.5)
        
        return {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'exposure': total_exposure,
            'diversification': diversification,
            'risk_score': risk_score
        }

    def _aggregate_signals(self, pred_time_signals, trade_time_signals, risk_signals, config):
        """Агрегация сигналов от всех модулей"""
        weights = config.get('weights', self.default_config['weights'])
        thresholds = config.get('thresholds', self.default_config['thresholds'])
        
        # Нормализация сигналов
        if pred_time_signals is not None:
            pred_score = np.mean(pred_time_signals) if isinstance(pred_time_signals, list) else pred_time_signals
        else:
            pred_score = 0.5
        
        if trade_time_signals is not None:
            if isinstance(trade_time_signals, dict):
                # Вероятности классов
                trade_score = trade_time_signals.get('buy', 0.33) - trade_time_signals.get('sell', 0.33)
            else:
                # Если trade_time_signals это список, берем среднее значение
                if isinstance(trade_time_signals, list):
                    trade_score = np.mean(trade_time_signals)
                else:
                    trade_score = trade_time_signals
        else:
            trade_score = 0.0
        
        if risk_signals is not None:
            if isinstance(risk_signals, dict):
                risk_score = 1.0 - risk_signals.get('risk_score', 0.5)  # Инвертируем риск
                volume_score = risk_signals.get('volume_score', 0.5)
            else:
                # Если risk_signals это список, берем среднее значение
                if isinstance(risk_signals, list):
                    risk_score = 1.0 - np.mean(risk_signals)
                else:
                    risk_score = 1.0 - risk_signals
                volume_score = 0.5
        else:
            risk_score = 0.5
            volume_score = 0.5
        
        # Взвешенная агрегация
        aggregated_score = (
            weights['pred_time'] * pred_score +
            weights['trade_time'] * trade_score +
            weights['risk'] * risk_score
        )
        
        # Принятие решения
        if aggregated_score > thresholds['buy_threshold']:
            decision = 'buy'
        elif aggregated_score < thresholds['sell_threshold']:
            decision = 'sell'
        else:
            decision = 'hold'
        
        # Расчет объема позиции
        position_size = volume_score * config.get('risk_limits', {}).get('max_position_size', 0.1)
        
        return {
            'decision': decision,
            'confidence': abs(aggregated_score - 0.5) * 2,  # 0-1
            'aggregated_score': aggregated_score,
            'position_size': position_size,
            'signals': {
                'pred_time': pred_score,
                'trade_time': trade_score,
                'risk': risk_score,
                'volume': volume_score
            }
        }

    def _apply_risk_management(self, decision, position_size, portfolio_metrics, config):
        """Применение риск-менеджмента"""
        risk_limits = config.get('risk_limits', self.default_config['risk_limits'])
        
        # Проверка лимитов
        if portfolio_metrics['exposure'] / portfolio_metrics['total_value'] > 0.8:
            position_size *= 0.5  # Уменьшаем размер позиции
        
        if portfolio_metrics['risk_score'] > 0.7:
            position_size *= 0.3  # Сильно уменьшаем при высоком риске
        
        # Проверка максимального размера позиции
        max_size = risk_limits.get('max_position_size', 0.1)
        position_size = min(position_size, max_size)
        
        # Добавляем стоп-лосс и тейк-профит
        stop_loss = risk_limits.get('stop_loss_pct', 0.05)
        take_profit = risk_limits.get('take_profit_pct', 0.15)
        
        return {
            'adjusted_signal': decision,  # Добавляем adjusted_signal
            'position_size': position_size,
            'stop_loss_pct': stop_loss,
            'take_profit_pct': take_profit,
            'max_leverage': risk_limits.get('max_leverage', 3.0)
        }

    def _create_ml_model(self, config):
        """Создание ML модели для агрегации (опционально)"""
        if config.get('mode') != 'ml':
            return None
        
        try:
            if xgb:
                model = xgb.XGBRegressor(
                    n_estimators=config.get('n_estimators', 100),
                    learning_rate=config.get('learning_rate', 0.1),
                    max_depth=config.get('max_depth', 6),
                    random_state=42
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=config.get('n_estimators', 100),
                    max_depth=config.get('max_depth', 6),
                    random_state=42
                )
            return model
        except Exception as e:
            logger.error(f"Error creating ML model: {e}")
            return None

    def train_model(self, coin_id, start_date, end_date, extra_config=None):
        """Обучение Trade Aggregator модели"""
        try:
            config = {**self.default_config, **(extra_config or {})}
            
            # Получение данных
            coin_data = orm_get_coin_data(coin_id, start_date, end_date)
            if not coin_data:
                raise ValueError(f"No data found for coin {coin_id}")
            
            # Получение новостного фона
            news_background = orm_get_news_background(coin_id, start_date, end_date)
            
            # Подготовка данных для обучения
            features = []
            targets = []
            
            for i in range(len(coin_data) - 1):
                # Технические индикаторы
                tech_features = calculate_technical_indicators(coin_data[:i+1])
                
                # Новостной фон
                news_score = news_background[i] if i < len(news_background) else 0.5
                
                # Ценовые изменения
                price_change = (coin_data[i+1]['close'] - coin_data[i]['close']) / coin_data[i]['close']
                
                # Целевая переменная (доходность)
                target = 1.0 if price_change > 0.01 else (-1.0 if price_change < -0.01 else 0.0)
                
                features.append({
                    'technical': tech_features,
                    'news': news_score,
                    'volume': coin_data[i]['volume'],
                    'volatility': abs(price_change)
                })
                targets.append(target)
            
            if len(features) < 100:
                raise ValueError("Insufficient data for training")
            
            # Создание и обучение модели (если включен ML режим)
            model = None
            if config.get('mode') == 'ml':
                model = self._create_ml_model(config)
                if model:
                    # Преобразование признаков в числовой формат
                    X = []
                    for f in features:
                        feature_vector = []
                        if isinstance(f['technical'], dict):
                            feature_vector.extend(f['technical'].values())
                        else:
                            feature_vector.append(f['technical'])
                        feature_vector.extend([f['news'], f['volume'], f['volatility']])
                        X.append(feature_vector)
                    
                    X = np.array(X)
                    y = np.array(targets)
                    
                    # Обучение
                    model.fit(X, y)
            
            # Сохранение модели и конфигурации
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.models_dir, f"trade_aggregator_{coin_id}_{timestamp}")
            
            os.makedirs(model_path, exist_ok=True)
            
            # Сохранение конфигурации
            config_path = os.path.join(model_path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Сохранение ML модели (если есть)
            if model:
                model_path_file = os.path.join(model_path, "model.pkl")
                with open(model_path_file, 'wb') as f:
                    pickle.dump(model, f)
            
            # Метаданные
            metadata = {
                'coin_id': coin_id,
                'start_date': start_date,
                'end_date': end_date,
                'config': config,
                'features_count': len(features),
                'model_type': config.get('mode'),
                'created_at': timestamp,
                'artifact_path': model_path
            }
            
            metadata_path = os.path.join(model_path, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Trade Aggregator model trained successfully for coin {coin_id}")
            
            return {
                'success': True,
                'model_path': model_path,
                'metadata': metadata,
                'config': config
            }
            
        except Exception as e:
            logger.error(f"Error training Trade Aggregator model: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def evaluate_model(self, coin_id, start_date, end_date, extra_config=None):
        """Оценка Trade Aggregator модели"""
        try:
            config = {**self.default_config, **(extra_config or {})}
            
            # Получение данных
            coin_data = orm_get_coin_data(coin_id, start_date, end_date)
            if not coin_data:
                raise ValueError(f"No data found for coin {coin_id}")
            
            # Симуляция торговли
            initial_balance = 10000  # $10,000
            balance = initial_balance
            positions = []
            trades = []
            
            for i in range(len(coin_data) - 1):
                # Получение сигналов (заглушка для демонстрации)
                pred_signal = np.random.normal(0.5, 0.2)  # Заглушка
                trade_signal = {'buy': 0.4, 'sell': 0.3, 'hold': 0.3}  # Заглушка
                risk_signal = {'risk_score': 0.3, 'volume_score': 0.6}  # Заглушка
                
                # Агрегация сигналов
                decision = self._aggregate_signals(
                    pred_signal, trade_signal, risk_signal, config
                )
                
                # Расчет метрик портфеля
                portfolio_metrics = self._calculate_portfolio_metrics(positions, balance)
                
                # Применение риск-менеджмента
                risk_managed = self._apply_risk_management(
                    decision['decision'], decision['position_size'], portfolio_metrics, config
                )
                
                # Исполнение сделки
                if decision['decision'] == 'buy' and decision['confidence'] > 0.6:
                    position_size = risk_managed['position_size'] * balance
                    entry_price = coin_data[i]['close']
                    
                    positions.append({
                        'size': position_size / entry_price,
                        'entry_price': entry_price,
                        'entry_time': coin_data[i]['timestamp']
                    })
                    
                    balance -= position_size
                    
                    trades.append({
                        'type': 'buy',
                        'price': entry_price,
                        'size': position_size,
                        'timestamp': coin_data[i]['timestamp']
                    })
                
                elif decision['decision'] == 'sell' and positions:
                    # Закрытие позиций
                    for pos in positions:
                        exit_price = coin_data[i]['close']
                        pnl = (exit_price - pos['entry_price']) * pos['size']
                        balance += (pos['size'] * exit_price) + pnl
                        
                        trades.append({
                            'type': 'sell',
                            'price': exit_price,
                            'size': pos['size'] * exit_price,
                            'pnl': pnl,
                            'timestamp': coin_data[i]['timestamp']
                        })
                    
                    positions = []
                
                # Обновление PnL для открытых позиций
                for pos in positions:
                    current_price = coin_data[i]['close']
                    pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['size']
            
            # Финальные метрики
            final_balance = balance
            for pos in positions:
                final_balance += pos['size'] * coin_data[-1]['close']
            
            total_return = (final_balance - initial_balance) / initial_balance
            win_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            total_trades = len(trades)
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            # Расчет Sharpe ratio (упрощенный)
            returns = []
            for i in range(1, len(coin_data)):
                ret = (coin_data[i]['close'] - coin_data[i-1]['close']) / coin_data[i-1]['close']
                returns.append(ret)
            
            if returns:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe = 0
            
            metrics = {
                'total_return_pct': total_return * 100,
                'final_balance': final_balance,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe,
                'max_drawdown': self._calculate_max_drawdown(coin_data),
                'volatility': np.std(returns) * np.sqrt(252) if returns else 0
            }
            
            logger.info(f"Trade Aggregator evaluation completed for coin {coin_id}")
            
            return {
                'success': True,
                'metrics': metrics,
                'trades': trades,
                'positions': positions,
                'config': config
            }
            
        except Exception as e:
            logger.error(f"Error evaluating Trade Aggregator model: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_max_drawdown(self, coin_data):
        """Расчет максимальной просадки"""
        if len(coin_data) < 2:
            return 0.0
        
        prices = [d['close'] for d in coin_data]
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            max_dd = max(max_dd, dd)
        
        return -max_dd  # Возвращаем отрицательное значение для просадки

    def predict(self, coin_id, pred_time_signals=None, trade_time_signals=None, 
                risk_signals=None, portfolio_state=None, extra_config=None):
        """Предсказание торгового решения"""
        try:
            config = {**self.default_config, **(extra_config or {})}
            
            # Агрегация сигналов
            decision = self._aggregate_signals(
                pred_time_signals, trade_time_signals, risk_signals, config
            )
            
            # Расчет метрик портфеля
            portfolio_metrics = self._calculate_portfolio_metrics(
                portfolio_state.get('positions', []) if portfolio_state else [],
                portfolio_state.get('balance', 10000) if portfolio_state else 10000
            )
            
            # Применение риск-менеджмента
            risk_managed = self._apply_risk_management(
                decision['decision'], decision['position_size'], portfolio_metrics, config
            )
            
            return {
                'decision': decision['decision'],
                'confidence': decision['confidence'],
                'position_size': risk_managed['position_size'],
                'stop_loss_pct': risk_managed['stop_loss_pct'],
                'take_profit_pct': risk_managed['take_profit_pct'],
                'max_leverage': risk_managed['max_leverage'],
                'signals': decision['signals'],
                'portfolio_metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in Trade Aggregator prediction: {e}")
            return {
                'decision': 'hold',
                'confidence': 0.0,
                'error': str(e)
            }

    def _save_model(self, model, config, metadata, model_path):
        """Сохранение модели"""
        try:
            os.makedirs(model_path, exist_ok=True)
            
            # Сохранение конфигурации
            config_path = os.path.join(model_path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Сохранение ML модели (если есть)
            if model:
                model_path_file = os.path.join(model_path, "model.pkl")
                with open(model_path_file, 'wb') as f:
                    pickle.dump(model, f)
            
            # Сохранение метаданных
            metadata_path = os.path.join(model_path, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, model_path):
        """Загрузка модели"""
        try:
            # Загрузка конфигурации
            config_path = os.path.join(model_path, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Загрузка ML модели (если есть)
            model = None
            model_file = os.path.join(model_path, "model.pkl")
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            
            # Загрузка метаданных
            metadata_path = os.path.join(model_path, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                'config': config,
                'model': model,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
