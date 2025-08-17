import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

try:
    import lightgbm as lgb
    import catboost as cb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import roc_auc_score, precision_recall_auc_score, confusion_matrix, classification_report
    from sklearn.ensemble import RandomForestClassifier
except ImportError as e:
    logging.warning(f"ML libraries not available: {e}")
    lgb = None
    cb = None

from ..database.orm.market import orm_get_coin_data
from ..database.orm.news import orm_get_news_background
from ..utils.metrics import calculate_technical_indicators

logger = logging.getLogger(__name__)

class TradeTimeService:
    """Сервис для Trade_time модуля - генерация торговых сигналов"""
    
    def __init__(self):
        self.models_dir = "models/models_pth/AgentTradeTime"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет технических индикаторов"""
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
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_20'] = df['close'].pct_change(periods=20)
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        return df
    
    def _prepare_features(self, df, news_data=None):
        """Подготовка признаков для модели"""
        # Технические индикаторы
        df = self._calculate_technical_indicators(df)
        
        # Создание целевой переменной (сигнал на следующий период)
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        
        # Определение сигналов на основе будущей доходности
        threshold = 0.02  # 2% порог
        df['signal'] = 0  # Hold
        df.loc[df['future_return'] > threshold, 'signal'] = 1  # Buy
        df.loc[df['future_return'] < -threshold, 'signal'] = -1  # Sell
        
        # Дополнительные признаки
        df['trend'] = np.where(df['sma_5'] > df['sma_20'], 1, -1)
        df['rsi_signal'] = np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))
        df['macd_signal_binary'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['bb_signal'] = np.where(df['close'] > df['bb_upper'], -1, np.where(df['close'] < df['bb_lower'], 1, 0))
        
        # Новостные данные
        if news_data is not None and not news_data.empty:
            df = df.merge(news_data[['timestamp', 'score', 'source_count']], 
                         left_on='timestamp', right_on='timestamp', how='left')
            df['news_score'] = df['score'].fillna(0)
            df['news_source_count'] = df['source_count'].fillna(0)
        else:
            df['news_score'] = 0
            df['news_source_count'] = 0
        
        # Удаление NaN значений
        df = df.dropna()
        
        # Выбор признаков для модели
        feature_columns = [
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'bb_position', 'volume_ratio', 'price_change', 'price_change_5', 'price_change_20',
            'volatility_ratio', 'trend', 'rsi_signal', 'macd_signal_binary', 'bb_signal',
            'news_score', 'news_source_count'
        ]
        
        X = df[feature_columns]
        y = df['signal']
        
        return X, y
    
    def _create_model(self, model_type: str = 'lightgbm', **kwargs) -> Any:
        """Создание модели указанного типа"""
        if model_type == 'lightgbm' and lgb is not None:
            return lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 6),
                num_leaves=kwargs.get('num_leaves', 31),
                random_state=42,
                verbose=-1
            )
        elif model_type == 'catboost' and cb is not None:
            return cb.CatBoostClassifier(
                iterations=kwargs.get('iterations', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                depth=kwargs.get('depth', 6),
                random_seed=42,
                verbose=False
            )
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_model(self, coin_id, start_date, end_date, 
                   model_type='lightgbm', extra_config=None):
        """Обучение модели Trade_time"""
        try:
            # Получение данных
            df = orm_get_coin_data(coin_id, start_date, end_date)
            if df.empty:
                raise ValueError(f"No data found for coin {coin_id}")
            
            # Получение новостных данных
            news_data = None
            try:
                news_data = orm_get_news_background(coin_id, start_date, end_date)
            except Exception as e:
                logger.warning(f"Could not fetch news data: {e}")
            
            # Подготовка признаков
            X, y = self._prepare_features(df, news_data)
            
            if len(X) < 100:
                raise ValueError(f"Insufficient data for training: {len(X)} samples")
            
            # Разделение данных
            test_size = extra_config.get('test_split', 0.2) if extra_config else 0.2
            val_size = extra_config.get('val_split', 0.2) if extra_config else 0.2
            
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp)
            
            # Создание и обучение модели
            model = self._create_model(model_type, **(extra_config or {}))
            
            # Обучение
            if model_type == 'catboost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)
            else:
                model.fit(X_train, y_train)
            
            # Оценка на валидационной выборке
            val_predictions = model.predict(X_val)
            val_probabilities = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
            
            # Метрики
            metrics = self._calculate_metrics(y_val, val_predictions, val_probabilities)
            
            # Сохранение модели
            model_path = self._save_model(model, coin_id, model_type, extra_config, metrics)
            
            return {
                'status': 'success',
                'model_path': model_path,
                'metrics': metrics,
                'data_info': {
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'feature_count': len(X.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error training Trade_time model: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Расчет метрик для классификации"""
        metrics = {}
        
        # Базовые метрики
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Точность по классам
        metrics['accuracy'] = (y_true == y_pred).mean()
        
        # Подсчет классов
        unique_labels = np.unique(y_true)
        for label in unique_labels:
            mask = y_true == label
            metrics[f'class_{label}_count'] = int(mask.sum())
            metrics[f'class_{label}_accuracy'] = (y_pred[mask] == y_true[mask]).mean() if mask.sum() > 0 else 0
        
        # ROC-AUC и PR-AUC если есть вероятности
        if y_prob is not None and len(unique_labels) > 1:
            try:
                # Для многоклассовой классификации используем one-vs-rest
                if len(unique_labels) > 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                    metrics['pr_auc'] = precision_recall_auc_score(y_true, y_prob, average='weighted')
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['pr_auc'] = precision_recall_auc_score(y_true, y_prob[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
                metrics['roc_auc'] = 0
                metrics['pr_auc'] = 0
        
        # Дополнительные метрики
        metrics['total_predictions'] = len(y_pred)
        metrics['buy_signals'] = int((y_pred == 1).sum())
        metrics['sell_signals'] = int((y_pred == -1).sum())
        metrics['hold_signals'] = int((y_pred == 0).sum())
        
        return metrics
    
    def _save_model(self, model, coin_id, model_type, 
                   extra_config, metrics):
        """Сохранение модели и артефактов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.models_dir, f"{coin_id}_{model_type}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Сохранение модели
        if model_type == 'lightgbm':
            model_path = os.path.join(model_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif model_type == 'catboost':
            model_path = os.path.join(model_dir, 'model.cbm')
            model.save_model(model_path)
        else:
            model_path = os.path.join(model_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Сохранение конфигурации
        config = {
            'coin_id': coin_id,
            'model_type': model_type,
            'timestamp': timestamp,
            'extra_config': extra_config,
            'metrics': metrics
        }
        
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Сохранение метаданных
        metadata = {
            'model_path': model_path,
            'config_path': config_path,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return model_dir
    
    def load_model(self, model_path: str):
        """Загрузка сохраненной модели"""
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_type = config['model_type']
        
        if model_type == 'lightgbm':
            model_file = os.path.join(model_path, 'model.pkl')
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        elif model_type == 'catboost':
            model_file = os.path.join(model_path, 'model.cbm')
            return cb.CatBoostClassifier().load_model(model_file)
        else:
            model_file = os.path.join(model_path, 'model.pkl')
            with open(model_file, 'rb') as f:
                return pickle.load(f)
    
    def predict(self, model_path, coin_id, start_date, end_date):
        """Предсказание торговых сигналов"""
        try:
            # Загрузка модели
            model = self.load_model(model_path)
            
            # Получение данных
            df = orm_get_coin_data(coin_id, start_date, end_date)
            if df.empty:
                raise ValueError(f"No data found for coin {coin_id}")
            
            # Получение новостных данных
            news_data = None
            try:
                news_data = orm_get_news_background(coin_id, start_date, end_date)
            except Exception as e:
                logger.warning(f"Could not fetch news data: {e}")
            
            # Подготовка признаков
            X, _ = self._prepare_features(df, news_data)
            
            # Предсказание
            predictions = model.predict(X)
            probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            # Формирование результата
            result = {
                'timestamp': df['timestamp'].tolist(),
                'predictions': predictions.tolist(),
                'signals': []
            }
            
            if probabilities is not None:
                result['probabilities'] = probabilities.tolist()
            
            # Преобразование предсказаний в сигналы
            for pred in predictions:
                if pred == 1:
                    result['signals'].append('BUY')
                elif pred == -1:
                    result['signals'].append('SELL')
                else:
                    result['signals'].append('HOLD')
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting with Trade_time model: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
