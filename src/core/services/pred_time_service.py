"""
Pred_time Service for price prediction using PyTorch models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from sqlalchemy.ext.asyncio import AsyncSession
import os
import json
import pickle
from pathlib import Path

from core.database.orm.market import orm_get_timeseries_by_coin, orm_get_data_timeseries
from core.services.news_background_service import NewsBackgroundService

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final layer
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out


class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, output_size: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(0):
            x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Take the last output and project to output size
        last_output = x[:, -1, :]
        out = self.dropout(last_output)
        out = self.output_projection(out)
        
        return out


class PredTimeService:
    """Service for Pred_time model training and inference"""
    
    def __init__(self, models_dir: str = "models/pred_time"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.news_service = NewsBackgroundService()
    
    def _get_news_background_for_window(self, coin_id: int, start_time: pd.Timestamp, end_time: pd.Timestamp) -> List[Dict[str, Any]]:
        """Get news background data for a specific time window"""
        try:
            # Convert pandas timestamps to datetime objects
            start_dt = start_time.to_pydatetime()
            end_dt = end_time.to_pydatetime()
            
            # Use synchronous wrapper for news service
            async def get_news_async():
                from core.database import db_helper
                async with db_helper.get_session() as session:
                    return await self.news_service.get_background_history(
                        session=session,
                        coin_id=coin_id,
                        start_time=start_dt,
                        end_time=end_dt,
                        limit=100
                    )
            
            # Run async function synchronously
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, use asyncio.create_task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, get_news_async())
                        return future.result()
                else:
                    return asyncio.run(get_news_async())
            except RuntimeError:
                # No event loop, create new one
                return asyncio.run(get_news_async())
                
        except Exception as e:
            logger.debug(f"Failed to get news background for window {start_time} - {end_time}: {e}")
            return []
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset"""
        df = df.copy()
        
        # Simple Moving Averages
        if "SMA" in self.config.get('technical_indicators', []):
            df['SMA_5'] = df['close'].rolling(window=5).mean()
            df['SMA_10'] = df['close'].rolling(window=10).mean()
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        if "RSI" in self.config.get('technical_indicators', []):
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        if "MACD" in self.config.get('technical_indicators', []):
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        if "BB" in self.config.get('technical_indicators', []):
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_10'] = df['close'].pct_change(periods=10)
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame, coin_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        # Calculate technical indicators
        df = self._calculate_technical_indicators(df)
        
        # Add news background features if enabled
        if self.config.get('news_integration', True):
            try:
                # Get news background for the coin
                news_scores = []
                for timestamp in df.index:
                    try:
                        # Try to get news background for this timestamp
                        # Use proper time alignment with window-based approach
                        window_start = timestamp - pd.Timedelta(hours=24)  # 24-hour window
                        window_end = timestamp
                        
                        # Get news background data for the time window
                        news_data = self._get_news_background_for_window(coin_id, window_start, window_end)
                        
                        if news_data and len(news_data) > 0:
                            # Calculate average score for the window
                            avg_score = np.mean([item.get('score', 0.0) for item in news_data])
                            news_scores.append(avg_score)
                        else:
                            news_scores.append(0.0)  # No news data available
                            
                    except Exception as e:
                        logger.debug(f"Failed to get news for timestamp {timestamp}: {e}")
                        news_scores.append(0.0)  # Fallback to neutral score
                
                df['news_score'] = news_scores
                
            except Exception as e:
                logger.warning(f"Failed to integrate news features for coin {coin_id}: {e}")
                df['news_score'] = 0.0
        
        # Select feature columns
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Add technical indicators
        for col in df.columns:
            if col.startswith(('SMA_', 'RSI', 'MACD', 'BB_', 'price_change', 'volume_')):
                feature_columns.append(col)
        
        # Add news score if available
        if 'news_score' in df.columns:
            feature_columns.append('news_score')
        
        # Remove rows with NaN values
        df_clean = df[feature_columns].dropna()
        
        if len(df_clean) < self.config['seq_len'] + self.config['pred_len']:
            raise ValueError(f"Insufficient data: {len(df_clean)} samples, need at least {self.config['seq_len'] + self.config['pred_len']}")
        
        # Prepare sequences
        X, y = [], []
        for i in range(len(df_clean) - self.config['seq_len'] - self.config['pred_len'] + 1):
            # Input sequence
            seq = df_clean.iloc[i:i + self.config['seq_len']][feature_columns[:-1]].values  # Exclude target
            
            # Target: future price changes
            target = df_clean.iloc[i + self.config['seq_len']:i + self.config['seq_len'] + self.config['pred_len']]['close'].values
            
            X.append(seq)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # Feature scaling
        if self.config['feature_scaling'] == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
        elif self.config['feature_scaling'] == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
        
        # Save scaler for inference
        self.scaler = scaler
        
        return X, y
    
    def _create_model(self) -> nn.Module:
        """Create the neural network model based on configuration"""
        input_size = self.feature_size
        output_size = self.config['pred_len']
        
        if self.config['model_type'] in ['LSTM', 'GRU']:
            if self.config['model_type'] == 'LSTM':
                model = LSTMModel(
                    input_size=input_size,
                    hidden_size=self.config['hidden_size'],
                    num_layers=self.config['num_layers'],
                    output_size=output_size,
                    dropout=self.config['dropout']
                )
            else:  # GRU
                model = nn.GRU(
                    input_size=input_size,
                    hidden_size=self.config['hidden_size'],
                    num_layers=self.config['num_layers'],
                    dropout=self.config['dropout'] if self.config['num_layers'] > 1 else 0,
                    batch_first=True
                )
                # Add output projection for GRU
                model = nn.Sequential(
                    model,
                    nn.Dropout(self.config['dropout']),
                    nn.Linear(self.config['hidden_size'], output_size)
                )
        
        elif self.config['model_type'] == 'Transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=self.config['d_model'],
                n_heads=self.config['n_heads'],
                n_layers=self.config['n_layers'],
                d_ff=self.config['d_ff'],
                output_size=output_size,
                dropout=self.config['dropout']
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.config['model_type']}")
        
        return model.to(self.device)
    
    async def train_model(
        self,
        session: AsyncSession,
        coin_ids: List[int],
        config: Dict[str, Any],
        agent_id: int
    ) -> Dict[str, Any]:
        """Train Pred_time model for specified coins"""
        try:
            self.config = config
            logger.info(f"Starting Pred_time model training for coins: {coin_ids}")
            
            # Collect data from all coins
            all_data = []
            for coin_id in coin_ids:
                try:
                    # Get timeseries data
                    ts = await orm_get_timeseries_by_coin(session, coin_id, timeframe='5m')
                    if not ts:
                        logger.warning(f"No timeseries found for coin {coin_id}")
                        continue
                    
                    data = await orm_get_data_timeseries(session, ts.id)
                    if not data:
                        logger.warning(f"No data found for timeseries {ts.id}")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame([
                        {
                            'timestamp': row.datetime,
                            'open': float(row.open),
                            'high': float(row.high),
                            'low': float(row.low),
                            'close': float(row.close),
                            'volume': float(row.volume)
                        }
                        for row in data
                    ])
                    df.set_index('timestamp', inplace=True)
                    df['coin_id'] = coin_id
                    
                    all_data.append(df)
                    
                except Exception as e:
                    logger.error(f"Failed to get data for coin {coin_id}: {e}")
                    continue
            
            if not all_data:
                raise ValueError("No data available for training")
            
            # Combine data from all coins
            combined_df = pd.concat(all_data, ignore_index=False)
            combined_df.sort_index(inplace=True)
            
            # Prepare features
            X, y = self._prepare_features(combined_df, coin_ids[0])
            self.feature_size = X.shape[-1]
            
            # Split data
            total_samples = len(X)
            test_size = int(total_samples * config['test_split'])
            val_size = int(total_samples * config['val_split'])
            train_size = total_samples - test_size - val_size
            
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
            X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
            
            logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False
            )
            
            # Create model
            model = self._create_model()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = []
            
            for epoch in range(config['epochs']):
                # Training
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                
                # Log progress
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                })
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{config['epochs']}: "
                              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Evaluate on test set
            model.eval()
            test_predictions = []
            test_targets = []
            
            with torch.no_grad():
                for i in range(0, len(X_test), config['batch_size']):
                    batch_X = torch.FloatTensor(X_test[i:i + config['batch_size']]).to(self.device)
                    batch_y = torch.FloatTensor(y_test[i:i + config['batch_size']]).to(self.device)
                    
                    outputs = model(batch_X)
                    test_predictions.extend(outputs.cpu().numpy())
                    test_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            test_predictions = np.array(test_predictions)
            test_targets = np.array(test_targets)
            
            metrics = self._calculate_metrics(test_predictions, test_targets)
            
            # Save model artifacts
            model_path = self._save_model(model, agent_id, config)
            
            return {
                'status': 'success',
                'model_path': model_path,
                'metrics': metrics,
                'training_history': training_history,
                'config': config,
                'data_info': {
                    'total_samples': total_samples,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'feature_size': self.feature_size
                }
            }
            
        except Exception as e:
            logger.exception("Pred_time model training failed")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # MAPE
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # Direction accuracy
        pred_direction = np.sign(predictions)
        target_direction = np.sign(targets)
        direction_accuracy = np.mean(pred_direction == target_direction)
        
        # Correlation
        correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'correlation': float(correlation),
            'r_squared': float(r_squared)
        }
    
    def _save_model(self, model: nn.Module, agent_id: int, config: Dict[str, Any]) -> str:
        """Save model artifacts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"agent_{agent_id}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), model_dir / "model.pth")
        
        # Save configuration
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        # Save scaler
        if hasattr(self, 'scaler'):
            with open(model_dir / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'agent_id': agent_id,
            'created_at': timestamp,
            'model_type': config['model_type'],
            'feature_size': getattr(self, 'feature_size', None),
            'device': str(self.device)
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(model_dir)
    
    async def load_model(self, model_path: str) -> Optional[nn.Module]:
        """Load a trained model"""
        try:
            model_dir = Path(model_path)
            
            # Load configuration
            with open(model_dir / "config.json", "r") as f:
                config = json.load(f)
            
            # Load metadata
            with open(model_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
            
            # Create model
            self.config = config
            self.feature_size = metadata['feature_size']
            model = self._create_model()
            
            # Load weights
            model.load_state_dict(torch.load(model_dir / "model.pth", map_location=self.device))
            
            # Load scaler
            if (model_dir / "scaler.pkl").exists():
                with open(model_dir / "scaler.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    async def predict(
        self,
        model: nn.Module,
        features: np.ndarray,
        coin_id: int
    ) -> Dict[str, Any]:
        """Make predictions using the trained model"""
        try:
            model.eval()
            
            with torch.no_grad():
                # Prepare input
                if len(features.shape) == 2:
                    features = features.reshape(1, -1, features.shape[-1])
                
                # Scale features if scaler is available
                if hasattr(self, 'scaler'):
                    features_reshaped = features.reshape(-1, features.shape[-1])
                    features_scaled = self.scaler.transform(features_reshaped)
                    features = features_scaled.reshape(features.shape)
                
                # Convert to tensor
                input_tensor = torch.FloatTensor(features).to(self.device)
                
                # Make prediction
                prediction = model(input_tensor)
                prediction = prediction.cpu().numpy()
            
            # Calculate direction
            if prediction[0] > 0:
                direction = "up"
            elif prediction[0] < 0:
                direction = "down"
            else:
                direction = "sideways"
            
            # Calculate confidence (simple approach - could be enhanced)
            confidence = min(abs(prediction[0]) / 0.1, 1.0)  # Normalize to [0, 1]
            
            return {
                'timestamp': datetime.now(),
                'coin_id': coin_id,
                'predicted_price': float(prediction[0]),
                'predicted_change': float(prediction[0]),
                'confidence': float(confidence),
                'direction': direction,
                'model_version': 'latest'
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
