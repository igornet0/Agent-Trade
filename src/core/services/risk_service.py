import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import math
import asyncio
from typing import Dict, Any, Tuple, Optional

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML libraries not available: {e}")
    xgb = None
    SKLEARN_AVAILABLE = False

from ..database.orm.market import orm_get_coin_data
from ..database.orm.news import orm_get_news_background
from ..database.engine import db_helper

logger = logging.getLogger(__name__)

def get_coin_data_sync(coin_id, start_date=None, end_date=None):
    """Синхронная обертка для orm_get_coin_data"""
    try:
        async def _get_data():
            async with db_helper.get_session() as session:
                return await orm_get_coin_data(session, coin_id, start_date, end_date)
        
        return asyncio.run(_get_data())
    except Exception as e:
        # Для тестов возвращаем пустой DataFrame если db_helper недоступен
        logger.warning(f"Database not available for testing: {e}")
        return pd.DataFrame()

class RiskService:
    """Сервис для Risk модуля - оценка рисков и объема торгов"""
    
    def __init__(self):
        self.models_dir = "models/models_pth/AgentRisk"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет технических индикаторов для риск-анализа"""
        df = df.copy()
        
        # Базовые индикаторы
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum and volatility
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_20'] = df['close'].pct_change(periods=20)
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Additional risk indicators
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['price_range_5'] = df['price_range'].rolling(window=5).mean()
        df['price_range_20'] = df['price_range'].rolling(window=20).mean()
        
        return df
    
    def _calculate_heuristic_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Расчет эвристического риска на основе технических индикаторов"""
        risk_score = pd.Series(0.0, index=df.index)
        
        # RSI risk (0-100, higher = more risk)
        rsi_risk = np.where(df['rsi'] > 70, 0.8, np.where(df['rsi'] < 30, 0.2, 0.5))
        risk_score += rsi_risk * 0.2
        
        # Volatility risk
        vol_risk = np.clip(df['volatility_ratio'] * 10, 0, 1)
        risk_score += vol_risk * 0.25
        
        # ATR risk
        atr_risk = np.clip(df['atr_ratio'] * 20, 0, 1)
        risk_score += atr_risk * 0.2
        
        # Price range risk
        range_risk = np.clip(df['price_range_5'] * 5, 0, 1)
        risk_score += range_risk * 0.15
        
        # Volume risk
        volume_risk = np.clip(df['volume_ratio'] * 0.5, 0, 1)
        risk_score += volume_risk * 0.1
        
        # Trend risk (trending markets are less risky)
        trend_risk = np.where(
            (df['sma_5'] > df['sma_20']) & (df['ema_12'] > df['ema_26']), 0.1,
            np.where(
                (df['sma_5'] < df['sma_20']) & (df['ema_12'] < df['ema_26']), 0.1, 0.3
            )
        )
        risk_score += trend_risk * 0.1
        
        return np.clip(risk_score, 0, 1)
    
    def _calculate_heuristic_volume_score(self, df: pd.DataFrame, risk_score: pd.Series) -> pd.Series:
        """Расчет эвристического объема на основе риска и других факторов"""
        volume_score = pd.Series(0.0, index=df.index)
        
        # Base volume from risk (inverse relationship)
        volume_score += (1 - risk_score) * 0.4
        
        # Volume from volatility (higher volatility = higher volume)
        vol_volume = np.clip(df['volatility_ratio'] * 5, 0, 1)
        volume_score += vol_volume * 0.3
        
        # Volume from trend strength
        trend_strength = np.abs(df['sma_5'] - df['sma_20']) / df['sma_20']
        trend_volume = np.clip(trend_strength * 10, 0, 1)
        volume_score += trend_volume * 0.2
        
        # Volume from news sentiment (if available)
        if 'news_score' in df.columns:
            news_volume = np.clip(np.abs(df['news_score']) * 2, 0, 1)
            volume_score += news_volume * 0.1
        
        return np.clip(volume_score, 0, 1)
    
    def _prepare_features(self, df, news_data=None):
        """Подготовка признаков для модели риска"""
        # Технические индикаторы
        df = self._calculate_technical_indicators(df)
        
        # Эвристические оценки
        risk_score = self._calculate_heuristic_risk_score(df)
        volume_score = self._calculate_heuristic_volume_score(df, risk_score)
        
        # Новостные данные
        if news_data is not None and not news_data.empty:
            df = df.merge(news_data[['timestamp', 'score', 'source_count']], 
                         left_on='timestamp', right_on='timestamp', how='left')
            df['news_score'] = df['score'].fillna(0)
            df['news_source_count'] = df['source_count'].fillna(0)
        else:
            df['news_score'] = 0
            df['news_source_count'] = 0
        
        # Целевые переменные
        df['risk_score'] = risk_score
        df['volume_score'] = volume_score
        
        # Дополнительные признаки для ML
        df['risk_momentum'] = risk_score.rolling(window=5).mean()
        df['risk_acceleration'] = risk_score.diff()
        df['volume_momentum'] = volume_score.rolling(window=5).mean()
        df['volume_acceleration'] = volume_score.diff()
        
        # Удаление NaN значений
        df = df.dropna()
        
        # Выбор признаков для модели
        feature_columns = [
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'bb_position', 'volume_ratio', 'price_change', 'price_change_5', 'price_change_20',
            'volatility_ratio', 'atr_ratio', 'price_range', 'price_range_5', 'price_range_20',
            'news_score', 'news_source_count', 'risk_momentum', 'risk_acceleration',
            'volume_momentum', 'volume_acceleration'
        ]
        
        X = df[feature_columns]
        y_risk = df['risk_score']
        y_volume = df['volume_score']
        
        return X, y_risk, y_volume
    
    def _create_model(self, model_type: str = 'xgboost', **kwargs) -> Any:
        """Создание модели указанного типа"""
        if model_type == 'xgboost' and xgb is not None:
            return xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                random_state=42,
                verbosity=0
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_model(self, coin_id, start_date, end_date, 
                   model_type='xgboost', extra_config=None):
        """Обучение модели Risk"""
        try:
            # Получение данных
            df = get_coin_data_sync(coin_id, start_date, end_date)
            if df.empty:
                raise ValueError(f"No data found for coin {coin_id}")
            
            # Получение новостных данных
            news_data = None
            try:
                news_data = orm_get_news_background(coin_id, start_date, end_date)
            except Exception as e:
                logger.warning(f"Could not fetch news data: {e}")
            
            # Подготовка признаков
            X, y_risk, y_volume = self._prepare_features(df, news_data)
            
            if len(X) < 100:
                raise ValueError(f"Insufficient data for training: {len(X)} samples")
            
            # Разделение данных
            test_size = extra_config.get('test_split', 0.2) if extra_config else 0.2
            val_size = extra_config.get('val_split', 0.2) if extra_config else 0.2
            
            # Для risk модели
            X_temp, X_test, y_risk_temp, y_risk_test = train_test_split(
                X, y_risk, test_size=test_size, random_state=42
            )
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_risk_train, y_risk_val = train_test_split(
                X_temp, y_risk_temp, test_size=val_size_adjusted, random_state=42
            )
            
            # Создание и обучение risk модели
            risk_model = self._create_model(model_type, **(extra_config or {}))
            risk_model.fit(X_train, y_risk_train)
            
            # Оценка risk модели
            risk_predictions = risk_model.predict(X_val)
            risk_metrics = self._calculate_risk_metrics(y_risk_val, risk_predictions)
            
            # Для volume модели
            X_temp, X_test, y_volume_temp, y_volume_test = train_test_split(
                X, y_volume, test_size=test_size, random_state=42
            )
            X_train, X_val, y_volume_train, y_volume_val = train_test_split(
                X_temp, y_volume_temp, test_size=val_size_adjusted, random_state=42
            )
            
            # Создание и обучение volume модели
            volume_model = self._create_model(model_type, **(extra_config or {}))
            volume_model.fit(X_train, y_volume_train)
            
            # Оценка volume модели
            volume_predictions = volume_model.predict(X_val)
            volume_metrics = self._calculate_volume_metrics(y_volume_val, volume_predictions)
            
            # Сохранение моделей
            model_path = self._save_models(risk_model, volume_model, coin_id, model_type, 
                                         extra_config, risk_metrics, volume_metrics)
            
            return {
                'status': 'success',
                'model_path': model_path,
                'risk_metrics': risk_metrics,
                'volume_metrics': volume_metrics,
                'data_info': {
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'feature_count': len(X.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error training Risk model: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_risk_metrics(self, y_true, y_pred):
        """Расчет метрик для risk модели"""
        metrics = {}
        
        if not SKLEARN_AVAILABLE:
            # Простые метрики без sklearn
            mse = np.mean((y_true - y_pred) ** 2)
            metrics['rmse'] = math.sqrt(mse)
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
        else:
            # Базовые метрики регрессии
            metrics['rmse'] = math.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
        
        # Дополнительные метрики
        metrics['mean_risk'] = float(np.mean(y_true))
        metrics['std_risk'] = float(np.std(y_true))
        metrics['max_risk'] = float(np.max(y_true))
        metrics['min_risk'] = float(np.min(y_true))
        
        # Корреляция
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics['r_squared'] = float(r2)
        
        return metrics
    
    def _calculate_volume_metrics(self, y_true, y_pred):
        """Расчет метрик для volume модели"""
        metrics = {}
        
        if not SKLEARN_AVAILABLE:
            # Простые метрики без sklearn
            mse = np.mean((y_true - y_pred) ** 2)
            metrics['rmse'] = math.sqrt(mse)
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
        else:
            # Базовые метрики регрессии
            metrics['rmse'] = math.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
        
        # Дополнительные метрики
        metrics['mean_volume'] = float(np.mean(y_true))
        metrics['std_volume'] = float(np.std(y_true))
        metrics['max_volume'] = float(np.max(y_true))
        metrics['min_volume'] = float(np.min(y_true))
        
        # Корреляция
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics['r_squared'] = float(r2)
        
        return metrics
    
    def _save_models(self, risk_model, volume_model, coin_id, model_type,
                    extra_config, risk_metrics, volume_metrics):
        """Сохранение моделей и артефактов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.models_dir, f"{coin_id}_{model_type}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Сохранение risk модели
        risk_model_path = os.path.join(model_dir, 'risk_model.pkl')
        with open(risk_model_path, 'wb') as f:
            pickle.dump(risk_model, f)
        
        # Сохранение volume модели
        volume_model_path = os.path.join(model_dir, 'volume_model.pkl')
        with open(volume_model_path, 'wb') as f:
            pickle.dump(volume_model, f)
        
        # Сохранение конфигурации
        config = {
            'coin_id': coin_id,
            'model_type': model_type,
            'timestamp': timestamp,
            'extra_config': extra_config,
            'risk_metrics': risk_metrics,
            'volume_metrics': volume_metrics
        }
        
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Сохранение метаданных
        metadata = {
            'risk_model_path': risk_model_path,
            'volume_model_path': volume_model_path,
            'config_path': config_path,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return model_dir
    
    def load_models(self, model_path: str):
        """Загрузка сохраненных моделей"""
        # Загрузка risk модели
        risk_model_path = os.path.join(model_path, 'risk_model.pkl')
        with open(risk_model_path, 'rb') as f:
            risk_model = pickle.load(f)
        
        # Загрузка volume модели
        volume_model_path = os.path.join(model_path, 'volume_model.pkl')
        with open(volume_model_path, 'rb') as f:
            volume_model = pickle.load(f)
        
        return risk_model, volume_model
    
    def predict(self, model_path, coin_id, start_date, end_date):
        """Предсказание рисков и объема"""
        try:
            # Загрузка моделей
            risk_model, volume_model = self.load_models(model_path)
            
            # Получение данных
            df = get_coin_data_sync(coin_id, start_date, end_date)
            if df.empty:
                raise ValueError(f"No data found for coin {coin_id}")
            
            # Получение новостных данных
            news_data = None
            try:
                news_data = orm_get_news_background(coin_id, start_date, end_date)
            except Exception as e:
                logger.warning(f"Could not fetch news data: {e}")
            
            # Подготовка признаков
            X, _, _ = self._prepare_features(df, news_data)
            
            # Предсказание
            risk_predictions = risk_model.predict(X)
            volume_predictions = volume_model.predict(X)
            
            # Формирование результата
            result = {
                'timestamp': df['timestamp'].tolist(),
                'risk_scores': np.clip(risk_predictions, 0, 1).tolist(),
                'volume_scores': np.clip(volume_predictions, 0, 1).tolist(),
                'risk_levels': [],
                'volume_levels': []
            }
            
            # Преобразование в уровни
            for risk in risk_predictions:
                if risk < 0.3:
                    result['risk_levels'].append('LOW')
                elif risk < 0.7:
                    result['risk_levels'].append('MEDIUM')
                else:
                    result['risk_levels'].append('HIGH')
            
            for volume in volume_predictions:
                if volume < 0.3:
                    result['volume_levels'].append('LOW')
                elif volume < 0.7:
                    result['volume_levels'].append('MEDIUM')
                else:
                    result['volume_levels'].append('HIGH')
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting with Risk model: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Расчет Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        # Убираем NaN значения
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return 0.0
        
        # Сортируем доходности
        sorted_returns = np.sort(returns)
        
        # Находим индекс для VaR
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var_index = max(0, min(var_index, len(sorted_returns) - 1))
        
        return float(sorted_returns[var_index])
    
    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        """Расчет Expected Shortfall (Conditional VaR)"""
        if len(returns) == 0:
            return 0.0
        
        # Убираем NaN значения
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            return 0.0
        
        # Находим VaR
        var = self.calculate_var(returns, confidence_level)
        
        # Находим среднее значение доходностей ниже VaR
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return float(np.mean(tail_returns))
