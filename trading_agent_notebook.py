# Система обучения агентов для торговли криптовалютой в реальном времени
# Копируйте этот код в ячейки Jupyter notebook

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Импортируем существующие модели ML
from backend.MMM.models.model_trade import TradingModel
from backend.MMM.models.model_ltsm import LTSMTimeFrame

# Класс для создания торговых меток
class TradingLabelGenerator:
    """Генератор торговых меток на основе будущих цен"""
    
    def __init__(self, profit_threshold: float = 0.02, stop_loss: float = 0.01):
        self.profit_threshold = profit_threshold
        self.stop_loss = stop_loss
    
    def generate_labels(self, data: pd.DataFrame, lookahead: int = 10) -> np.ndarray:
        """
        Генерирует торговые метки:
        0 - HOLD (держать)
        1 - BUY (покупать)
        2 - SELL (продавать)
        """
        labels = np.zeros(len(data))
        
        for i in range(len(data) - lookahead):
            current_price = data.iloc[i]['close']
            future_prices = data.iloc[i+1:i+lookahead+1]['close']
            
            # Максимальная прибыль и убыток
            max_profit = (future_prices.max() - current_price) / current_price
            max_loss = (future_prices.min() - current_price) / current_price
            
            if max_profit >= self.profit_threshold:
                labels[i] = 1  # BUY
            elif max_loss <= -self.stop_loss:
                labels[i] = 2  # SELL
            else:
                labels[i] = 0  # HOLD
                
        return labels

# Класс для подготовки данных
class CryptoDataset(Dataset):
    """Датасет для обучения торгового агента"""
    
    def __init__(self, data: pd.DataFrame, seq_len: int = 50, pred_len: int = 5, 
                 normalize: bool = True):
        self.data = data.copy()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.normalize = normalize
        
        # Генерируем метки
        label_generator = TradingLabelGenerator()
        self.labels = label_generator.generate_labels(data)
        
        # Подготавливаем признаки
        self.features = self._prepare_features()
        
        # Нормализация
        if normalize:
            self.features = self._normalize_features()
        
        # Убираем строки с недостаточными данными
        self.valid_indices = self._get_valid_indices()
        
    def _prepare_features(self) -> np.ndarray:
        """Подготавливает признаки для модели"""
        # Базовые признаки
        feature_cols = ['open', 'close', 'max', 'min', 'volume']
        features = self.data[feature_cols].values
        
        # Технические индикаторы
        features = self._add_technical_indicators(features)
        
        return features
    
    def _add_technical_indicators(self, features: np.ndarray) -> np.ndarray:
        """Добавляет технические индикаторы"""
        # Простые скользящие средние
        sma_5 = self.data['close'].rolling(5).mean().values
        sma_20 = self.data['close'].rolling(20).mean().values
        
        # RSI
        rsi = self._calculate_rsi(self.data['close']).values
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(self.data['close'])
        
        # MACD
        macd, signal = self._calculate_macd(self.data['close'])
        
        # Объединяем все признаки
        additional_features = np.column_stack([
            sma_5, sma_20, rsi, bb_upper, bb_lower, macd, signal
        ])
        
        # Заменяем NaN на 0
        additional_features = np.nan_to_num(additional_features, 0)
        
        return np.column_stack([features, additional_features])
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Вычисляет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Вычисляет Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Вычисляет MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line
    
    def _normalize_features(self) -> np.ndarray:
        """Нормализует признаки"""
        # Z-score нормализация
        mean = np.nanmean(self.features, axis=0)
        std = np.nanstd(self.features, axis=0)
        std[std == 0] = 1  # Избегаем деления на 0
        
        normalized = (self.features - mean) / std
        return np.nan_to_num(normalized, 0)
    
    def _get_valid_indices(self) -> List[int]:
        """Возвращает индексы валидных последовательностей"""
        valid_indices = []
        for i in range(self.seq_len, len(self.data) - self.pred_len):
            if not np.any(np.isnan(self.features[i-self.seq_len:i+self.pred_len])):
                valid_indices.append(i)
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Возвращает последовательность данных, временные метки и метки"""
        data_idx = self.valid_indices[idx]
        
        # Последовательность данных
        sequence = self.features[data_idx-self.seq_len:data_idx]
        sequence_tensor = torch.FloatTensor(sequence)
        
        # Временные метки
        time_data = self._extract_time_features(data_idx)
        time_tensor = torch.FloatTensor(time_data)
        
        # Метки
        label = self.labels[data_idx:data_idx+self.pred_len]
        label_tensor = torch.LongTensor(label)
        
        return sequence_tensor, time_tensor, label_tensor
    
    def _extract_time_features(self, idx: int) -> np.ndarray:
        """Извлекает временные признаки"""
        datetime_obj = self.data.iloc[idx]['datetime']
        
        # Месяц (0-11)
        month = datetime_obj.month - 1
        # День недели (0-6)
        weekday = datetime_obj.weekday()
        # Час (0-23)
        hour = datetime_obj.hour
        # Минута (0-59)
        minute = datetime_obj.minute
        # День месяца (1-31)
        day = datetime_obj.day
        
        return np.array([month, weekday, hour, minute, day])

# Класс для ансамбля моделей
class ModelEnsemble:
    """Ансамбль торговых моделей"""
    
    def __init__(self, input_size: int, device: torch.device):
        self.models = []
        self.device = device
        self.input_size = input_size
        self.weights = []
        
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """Добавляет модель в ансамбль"""
        model.to(self.device)
        self.models.append(model)
        self.weights.append(weight)
        
        # Нормализуем веса
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict(self, x: torch.Tensor, time_data: torch.Tensor) -> torch.Tensor:
        """Делает предсказание используя ансамбль"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(x, x, time_data)  # Используем x как x_pred для совместимости
                predictions.append(pred * weight)
        
        # Объединяем предсказания
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        return ensemble_pred
    
    def update_weights(self, performance_scores: List[float]):
        """Обновляет веса моделей на основе их производительности"""
        if len(performance_scores) != len(self.weights):
            return
            
        # Нормализуем оценки производительности
        total_score = sum(performance_scores)
        if total_score > 0:
            normalized_scores = [score / total_score for score in performance_scores]
            
            # Обновляем веса с учетом производительности
            for i in range(len(self.weights)):
                self.weights[i] = 0.7 * self.weights[i] + 0.3 * normalized_scores[i]
            
            # Нормализуем веса
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]

# Класс для адаптивного обучения
class AdaptiveLearningAgent:
    """Агент с адаптивным обучением на ошибках"""
    
    def __init__(self, coin_name: str, data: pd.DataFrame, device: torch.device):
        self.coin_name = coin_name
        self.data = data
        self.device = device
        
        # Параметры модели
        self.seq_len = 50
        self.pred_len = 5
        self.input_size = 12  # Базовые признаки + технические индикаторы
        
        # Создаем ансамбль моделей
        self.ensemble = ModelEnsemble(self.input_size, device)
        self._initialize_models()
        
        # Параметры обучения
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        
        # История торговли и обучения
        self.trading_history = []
        self.learning_history = []
        self.performance_metrics = []
        
        # Параметры торговли
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0
        self.stop_loss = 0.02
        self.take_profit = 0.04
        
    def _initialize_models(self):
        """Инициализирует несколько моделей с разными архитектурами"""
        # Модель 1: Базовая LSTM
        model1 = TradingModel(
            input_size=self.input_size,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            lstm_hidden=128,
            hidden_size=64,
            num_layers=2,
            dropout=0.3
        )
        
        # Модель 2: Глубокая LSTM
        model2 = TradingModel(
            input_size=self.input_size,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            lstm_hidden=256,
            hidden_size=128,
            num_layers=4,
            dropout=0.4
        )
        
        # Модель 3: Широкая LSTM
        model3 = TradingModel(
            input_size=self.input_size,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            lstm_hidden=512,
            hidden_size=256,
            num_layers=3,
            dropout=0.2
        )
        
        # Добавляем модели в ансамбль
        self.ensemble.add_model(model1, weight=0.4)
        self.ensemble.add_model(model2, weight=0.35)
        self.ensemble.add_model(model3, weight=0.25)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Подготавливает данные для обучения"""
        # Создаем датасет
        dataset = CryptoDataset(
            self.data, 
            seq_len=self.seq_len, 
            pred_len=self.pred_len
        )
        
        # Разделяем на train и validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Создаем DataLoader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_models(self, train_loader: DataLoader, val_loader: DataLoader):
        """Обучает все модели в ансамбле"""
        for i, model in enumerate(self.ensemble.models):
            print(f"Обучение модели {i+1}/{len(self.ensemble.models)}")
            
            # Оптимизатор и функция потерь
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Обучение
            for epoch in range(self.epochs):
                train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_accuracy = self._validate_epoch(model, val_loader, criterion)
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}: "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Accuracy: {val_accuracy:.4f}")
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                     optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Обучает модель на одной эпохе"""
        model.train()
        total_loss = 0
        
        for batch_idx, (sequence, time_data, labels) in enumerate(train_loader):
            sequence = sequence.to(self.device)
            time_data = time_data.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Предсказание
            outputs = model(sequence, sequence, time_data)
            
            # Вычисляем потери для каждого шага предсказания
            loss = 0
            for step in range(self.pred_len):
                loss += criterion(outputs[:, step, :], labels[:, step])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Валидирует модель на одной эпохе"""
        model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for sequence, time_data, labels in val_loader:
                sequence = sequence.to(self.device)
                time_data = time_data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(sequence, sequence, time_data)
                
                # Вычисляем потери
                loss = 0
                for step in range(self.pred_len):
                    loss += criterion(outputs[:, step, :], labels[:, step])
                    
                    # Точность предсказаний
                    _, predicted = torch.max(outputs[:, step, :], 1)
                    correct_predictions += (predicted == labels[:, step]).sum().item()
                    total_predictions += labels[:, step].size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return avg_loss, accuracy
    
    def make_trading_decision(self, current_data: pd.DataFrame) -> Dict[str, any]:
        """Принимает торговое решение на основе предсказаний"""
        # Подготавливаем данные для предсказания
        sequence = self._prepare_sequence(current_data)
        time_data = self._extract_time_features(current_data)
        
        # Делаем предсказание
        with torch.no_grad():
            prediction = self.ensemble.predict(sequence, time_data)
            probabilities = torch.softmax(prediction, dim=-1)
        
        # Анализируем предсказания
        action_probs = probabilities.mean(dim=0)  # Среднее по всем шагам
        action = torch.argmax(action_probs).item()
        confidence = action_probs.max().item()
        
        # Принимаем решение
        decision = self._analyze_trading_decision(action, confidence, current_data)
        
        return decision
    
    def _prepare_sequence(self, data: pd.DataFrame) -> torch.Tensor:
        """Подготавливает последовательность для предсказания"""
        # Берем последние seq_len строк
        recent_data = data.tail(self.seq_len)
        
        # Подготавливаем признаки (аналогично CryptoDataset)
        features = self._extract_features(recent_data)
        
        # Нормализация
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        sequence = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        return sequence
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Извлекает признаки из данных"""
        # Базовые признаки
        feature_cols = ['open', 'close', 'max', 'min', 'volume']
        features = data[feature_cols].values
        
        # Добавляем технические индикаторы (упрощенно)
        close_prices = data['close'].values
        
        # SMA
        sma_5 = np.convolve(close_prices, np.ones(5)/5, mode='same')
        sma_20 = np.convolve(close_prices, np.ones(20)/20, mode='same')
        
        # RSI (упрощенно)
        rsi = np.zeros_like(close_prices)
        for i in range(1, len(close_prices)):
            if close_prices[i-1] != 0:
                rsi[i] = 100 - (100 / (1 + (close_prices[i] - close_prices[i-1]) / close_prices[i-1]))
        
        # Объединяем признаки
        additional_features = np.column_stack([sma_5, sma_20, rsi])
        additional_features = np.nan_to_num(additional_features, 0)
        
        return np.column_stack([features, additional_features])
    
    def _extract_time_features(self, data: pd.DataFrame) -> torch.Tensor:
        """Извлекает временные признаки"""
        latest_time = data.iloc[-1]['datetime']
        
        time_features = np.array([
            latest_time.month - 1,
            latest_time.weekday(),
            latest_time.hour,
            latest_time.minute,
            latest_time.day
        ])
        
        return torch.FloatTensor(time_features).unsqueeze(0).unsqueeze(0).to(self.device)
    
    def _analyze_trading_decision(self, action: int, confidence: float, 
                                 current_data: pd.DataFrame) -> Dict[str, any]:
        """Анализирует торговое решение"""
        current_price = current_data.iloc[-1]['close']
        
        decision = {
            'action': ['HOLD', 'BUY', 'SELL'][action],
            'confidence': confidence,
            'current_price': current_price,
            'timestamp': current_data.iloc[-1]['datetime'],
            'reasoning': []
        }
        
        # Анализ текущей позиции
        if self.position == 0:  # Нет позиции
            if action == 1 and confidence > 0.7:  # BUY с высокой уверенностью
                decision['reasoning'].append("Высокая уверенность в росте цены")
                decision['recommended_action'] = 'BUY'
            elif action == 2 and confidence > 0.7:  # SELL с высокой уверенностью
                decision['reasoning'].append("Высокая уверенность в падении цены")
                decision['recommended_action'] = 'SELL'
            else:
                decision['reasoning'].append("Недостаточная уверенность для открытия позиции")
                decision['recommended_action'] = 'HOLD'
        
        elif self.position == 1:  # Длинная позиция
            if action == 2 and confidence > 0.6:  # SELL
                decision['reasoning'].append("Сигнал на продажу - закрываем длинную позицию")
                decision['recommended_action'] = 'SELL'
            else:
                decision['reasoning'].append("Удерживаем длинную позицию")
                decision['recommended_action'] = 'HOLD'
        
        elif self.position == -1:  # Короткая позиция
            if action == 1 and confidence > 0.6:  # BUY
                decision['reasoning'].append("Сигнал на покупку - закрываем короткую позицию")
                decision['recommended_action'] = 'BUY'
            else:
                decision['reasoning'].append("Удерживаем короткую позицию")
                decision['recommended_action'] = 'HOLD'
        
        return decision
    
    def execute_trade(self, decision: Dict[str, any]) -> Dict[str, any]:
        """Выполняет торговую операцию"""
        trade_result = {
            'timestamp': decision['timestamp'],
            'action': decision['recommended_action'],
            'price': decision['current_price'],
            'position_before': self.position,
            'position_after': self.position,
            'profit_loss': 0,
            'success': False
        }
        
        if decision['recommended_action'] == 'BUY':
            if self.position == 0:  # Открываем длинную позицию
                self.position = 1
                self.entry_price = decision['current_price']
                trade_result['position_after'] = 1
                trade_result['reasoning'] = "Открыта длинная позиция"
            elif self.position == -1:  # Закрываем короткую позицию
                profit_loss = self.entry_price - decision['current_price']
                self.position = 0
                trade_result['position_after'] = 0
                trade_result['profit_loss'] = profit_loss
                trade_result['success'] = profit_loss > 0
                trade_result['reasoning'] = f"Закрыта короткая позиция, P&L: {profit_loss:.4f}"
        
        elif decision['recommended_action'] == 'SELL':
            if self.position == 0:  # Открываем короткую позицию
                self.position = -1
                self.entry_price = decision['current_price']
                trade_result['position_after'] = -1
                trade_result['reasoning'] = "Открыта короткая позиция"
            elif self.position == 1:  # Закрываем длинную позицию
                profit_loss = decision['current_price'] - self.entry_price
                self.position = 0
                trade_result['position_after'] = 0
                trade_result['profit_loss'] = profit_loss
                trade_result['success'] = profit_loss > 0
                trade_result['reasoning'] = f"Закрыта длинная позиция, P&L: {profit_loss:.4f}"
        
        # Сохраняем результат торговли
        self.trading_history.append(trade_result)
        
        return trade_result
    
    def learn_from_trades(self):
        """Обучается на результатах торговли"""
        if len(self.trading_history) < 10:
            return
        
        # Анализируем последние сделки
        recent_trades = self.trading_history[-50:]
        success_rate = sum(1 for trade in recent_trades if trade['success']) / len(recent_trades)
        
        # Обновляем веса моделей на основе производительности
        if success_rate < 0.4:  # Низкая успешность
            # Увеличиваем вес лучших моделей
            self._adjust_model_weights(success_rate)
            
            # Переобучаем худшие модели
            self._retrain_underperforming_models()
        
        # Сохраняем метрики
        self.performance_metrics.append({
            'timestamp': datetime.now(),
            'success_rate': success_rate,
            'total_trades': len(self.trading_history),
            'recent_pnl': sum(trade['profit_loss'] for trade in recent_trades)
        })
    
    def _adjust_model_weights(self, success_rate: float):
        """Корректирует веса моделей"""
        # Упрощенная логика корректировки весов
        if success_rate < 0.3:
            # Увеличиваем вес первой модели (базовой)
            self.ensemble.weights[0] *= 1.1
            self.ensemble.weights[1] *= 0.95
            self.ensemble.weights[2] *= 0.95
            
            # Нормализуем веса
            total_weight = sum(self.ensemble.weights)
            self.ensemble.weights = [w / total_weight for w in self.ensemble.weights]
    
    def _retrain_underperforming_models(self):
        """Переобучает плохо работающие модели"""
        # Здесь можно добавить логику переобучения
        # Например, уменьшить learning rate или изменить архитектуру
        pass
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Возвращает сводку по производительности"""
        if not self.trading_history:
            return {"message": "Нет торговой истории"}
        
        total_trades = len(self.trading_history)
        successful_trades = sum(1 for trade in self.trading_history if trade['success'])
        total_pnl = sum(trade['profit_loss'] for trade in self.trading_history)
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'success_rate': successful_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'average_pnl': total_pnl / total_trades if total_trades > 0 else 0,
            'current_position': self.position,
            'model_weights': self.ensemble.weights
        }

# Функция для тестирования агента
def test_agent_trading(agent: AdaptiveLearningAgent, test_period_days: int = 7):
    """Тестирует агента на исторических данных"""
    print(f"Тестируем агента на {test_period_days} днях данных")
    
    # Берем последние данные для тестирования
    test_data = agent.data.tail(test_period_days * 288)  # 288 = 24*12 (5-минутные интервалы)
    
    results = []
    
    for i in range(len(test_data) - agent.seq_len):
        # Берем текущий срез данных
        current_slice = test_data.iloc[i:i+agent.seq_len]
        
        # Принимаем торговое решение
        decision = agent.make_trading_decision(current_slice)
        
        # Выполняем сделку
        trade_result = agent.execute_trade(decision)
        
        if trade_result['action'] != 'HOLD':
            results.append(trade_result)
            print(f"Сделка: {trade_result['action']} по цене {trade_result['price']:.4f}, "
                  f"P&L: {trade_result['profit_loss']:.4f}")
    
    # Анализируем результаты
    agent.learn_from_trades()
    performance = agent.get_performance_summary()
    
    print("\nРезультаты тестирования:")
    print(f"Всего сделок: {performance['total_trades']}")
    print(f"Успешных сделок: {performance['successful_trades']}")
    print(f"Процент успеха: {performance['success_rate']:.2%}")
    print(f"Общий P&L: {performance['total_pnl']:.4f}")
    print(f"Средний P&L: {performance['average_pnl']:.4f}")
    
    return results

# Функция для создания и обучения агента
def create_and_train_agent(coin_name: str, data: pd.DataFrame, device: torch.device):
    """Создает и обучает торгового агента для конкретной монеты"""
    print(f"Создаем и обучаем агента для {coin_name}")
    
    # Создаем агента
    agent = AdaptiveLearningAgent(coin_name, data, device)
    
    # Подготавливаем данные
    print("Подготавливаем данные для обучения...")
    train_loader, val_loader = agent.prepare_data()
    
    # Обучаем модели
    print("Обучаем модели...")
    agent.train_models(train_loader, val_loader)
    
    print(f"Обучение завершено для {coin_name}")
    return agent
