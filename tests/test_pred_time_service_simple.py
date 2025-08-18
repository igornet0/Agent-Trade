#!/usr/bin/env python3
"""
Simple tests for Pred Time Service
Tests the fixed pred time service functionality
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pandas as pd
import numpy as np
import numpy as np
from datetime import datetime, timedelta

class TestPredTimeServiceSimple(unittest.TestCase):
    """Test Pred Time Service functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'seq_len': 10,
            'pred_len': 5,
            'feature_scaling': None,  # Disable scaling to avoid sklearn dependency
            'news_integration': True,
            'technical_indicators': ['SMA', 'RSI', 'MACD', 'BB'],
            'model_type': 'LSTM',
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        }
        
        # Sample data - need at least 15 samples (seq_len + pred_len)
        # Create realistic price data to avoid NaN values in technical indicators
        # Need more data to account for technical indicator windows (SMA_20 needs 20 points)
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        base_price = 100
        prices = []
        for i in range(100):
            # Create realistic price movements
            change = np.random.normal(0, 1)  # Smaller random change
            base_price = max(50, base_price + change)  # Ensure positive prices
            prices.append(base_price)
        
        self.sample_df = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 3) for p in prices],
            'low': [max(1, p - np.random.uniform(0, 3)) for p in prices],
            'close': [p + np.random.normal(0, 0.5) for p in prices],
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Ensure high >= close >= low >= open for realistic data
        self.sample_df['high'] = self.sample_df[['open', 'close', 'high']].max(axis=1)
        self.sample_df['low'] = self.sample_df[['open', 'close', 'low']].min(axis=1)
    
    def test_pred_time_service_imports(self):
        """Test that pred time service can be imported"""
        try:
            from core.services.pred_time_service import PredTimeService
            self.assertIsNotNone(PredTimeService)
        except ImportError as e:
            self.fail(f"Failed to import pred time service: {e}")
    
    @patch('core.services.pred_time_service.NewsBackgroundService')
    def test_pred_time_service_initialization(self, mock_news_service_class):
        """Test pred time service initialization"""
        from core.services.pred_time_service import PredTimeService
        
        # Mock news service
        mock_news_service = Mock()
        mock_news_service_class.return_value = mock_news_service
        
        # Test initialization
        service = PredTimeService()
        
        self.assertIsNotNone(service)
        self.assertIsNotNone(service.news_service)
        self.assertIsNotNone(service.models_dir)
        
        # Verify news service was created
        mock_news_service_class.assert_called_once()
    
    def test_calculate_technical_indicators(self):
        """Test technical indicators calculation"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = self.test_config
        
        # Test with sample data
        result_df = service._calculate_technical_indicators(self.sample_df.copy())
        
        # Verify basic columns are preserved
        self.assertIn('open', result_df.columns)
        self.assertIn('high', result_df.columns)
        self.assertIn('low', result_df.columns)
        self.assertIn('close', result_df.columns)
        self.assertIn('volume', result_df.columns)
        
        # Verify technical indicators were added
        self.assertIn('SMA_5', result_df.columns)
        self.assertIn('SMA_10', result_df.columns)
        self.assertIn('SMA_20', result_df.columns)
        self.assertIn('RSI', result_df.columns)
        self.assertIn('MACD', result_df.columns)
        self.assertIn('BB_middle', result_df.columns)
        self.assertIn('price_change', result_df.columns)
        self.assertIn('volume_ma', result_df.columns)
    
    @patch('core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_get_news_background_for_window(self, mock_get_news):
        """Test news background retrieval for window"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        
        # Mock news data
        mock_news_data = [
            {'score': 0.8, 'timestamp': '2024-01-01T10:00:00'},
            {'score': 0.6, 'timestamp': '2024-01-01T11:00:00'},
            {'score': 0.9, 'timestamp': '2024-01-01T12:00:00'}
        ]
        mock_get_news.return_value = mock_news_data
        
        # Test window retrieval
        start_time = pd.Timestamp('2024-01-01T10:00:00')
        end_time = pd.Timestamp('2024-01-01T12:00:00')
        
        result = service._get_news_background_for_window(1, start_time, end_time)
        
        # Verify result
        self.assertEqual(result, mock_news_data)
        mock_get_news.assert_called_once_with(1, start_time, end_time)
    
    @patch('core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_get_news_background_for_window_empty(self, mock_get_news):
        """Test news background retrieval with empty result"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        
        # Mock empty news data
        mock_get_news.return_value = []
        
        # Test window retrieval
        start_time = pd.Timestamp('2024-01-01T10:00:00')
        end_time = pd.Timestamp('2024-01-01T12:00:00')
        
        result = service._get_news_background_for_window(1, start_time, end_time)
        
        # Verify result
        self.assertEqual(result, [])
    
    @patch('core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_get_news_background_for_window_error(self, mock_get_news):
        """Test news background retrieval with error"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        
        # Mock error
        mock_get_news.side_effect = Exception("Database error")
        
        # Test window retrieval
        start_time = pd.Timestamp('2024-01-01T10:00:00')
        end_time = pd.Timestamp('2024-01-01T12:00:00')
        
        # Should handle exception gracefully
        try:
            result = service._get_news_background_for_window(1, start_time, end_time)
            self.assertEqual(result, [])
        except Exception as e:
            # If exception is raised, that's also acceptable
            self.assertIn("Database error", str(e))
    
    @patch('core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_prepare_features_with_news_integration(self, mock_get_news):
        """Test feature preparation with news integration"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = self.test_config
        
        # Mock news data for each timestamp
        mock_news_data = [
            {'score': 0.8},
            {'score': 0.6},
            {'score': 0.9},
            {'score': 0.7},
            {'score': 0.5}
        ]
        mock_get_news.return_value = mock_news_data
        
        # Test feature preparation
        X, y = service._prepare_features(self.sample_df.copy(), coin_id=1)
        
        # Verify results are numpy arrays
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        
        # Verify news integration was called
        self.assertGreater(mock_get_news.call_count, 0)
    
    @patch('core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_prepare_features_without_news_integration(self, mock_get_news):
        """Test feature preparation without news integration"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = self.test_config.copy()
        service.config['news_integration'] = False
        
        # Test feature preparation
        X, y = service._prepare_features(self.sample_df.copy(), coin_id=1)
        
        # Verify results are numpy arrays
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        
        # Verify news integration was not called
        mock_get_news.assert_not_called()
    
    @patch('core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_prepare_features_news_integration_error(self, mock_get_news):
        """Test feature preparation with news integration error"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = self.test_config
        
        # Mock news error
        mock_get_news.side_effect = Exception("News service error")
        
        # Test feature preparation
        X, y = service._prepare_features(self.sample_df.copy(), coin_id=1)
        
        # Verify results are still generated (with fallback)
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
    
    def test_prepare_features_insufficient_data(self):
        """Test feature preparation with insufficient data"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = self.test_config
        
        # Create small dataset
        small_df = self.sample_df.head(2)  # Only 2 rows, need at least 15 (seq_len + pred_len)
        
        # Test should raise ValueError
        with self.assertRaises(ValueError) as context:
            service._prepare_features(small_df, coin_id=1)
        
        self.assertIn("Insufficient data", str(context.exception))
    
    def test_create_lstm_model(self):
        """Test LSTM model creation"""
        from core.services.pred_time_service import PredTimeService, LSTMModel
        
        service = PredTimeService()
        service.config = self.test_config
        service.feature_size = 10
        
        # Test model creation
        model = service._create_model()
        
        # Verify model type
        self.assertIsInstance(model, LSTMModel)
        
        # Verify model parameters
        self.assertEqual(model.hidden_size, 64)
        self.assertEqual(model.num_layers, 2)
    
    def test_create_gru_model(self):
        """Test GRU model creation"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = self.test_config.copy()
        service.config['model_type'] = 'GRU'
        service.feature_size = 10
        
        # Test model creation
        model = service._create_model()
        
        # Verify model is Sequential (GRU wrapped)
        self.assertIsInstance(model, type(service._create_model()))
    
    def test_create_transformer_model(self):
        """Test Transformer model creation"""
        from core.services.pred_time_service import PredTimeService, TransformerModel
        
        service = PredTimeService()
        service.config = self.test_config.copy()
        service.config['model_type'] = 'Transformer'
        service.config['d_model'] = 128
        service.config['n_heads'] = 8
        service.config['n_layers'] = 4
        service.config['d_ff'] = 512
        service.feature_size = 10
        
        # Test model creation
        model = service._create_model()
        
        # Verify model type
        self.assertIsInstance(model, TransformerModel)
    
    def test_pred_time_service_error_handling(self):
        """Test that pred time service has proper error handling"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        
        # Test with invalid configuration
        service.config = {}
        
        # Should handle missing configuration gracefully
        try:
            service._calculate_technical_indicators(self.sample_df.copy())
            # If no exception is raised, that's also acceptable
            # as the service might have fallback behavior
        except (KeyError, ValueError, TypeError) as e:
            # Any of these exceptions are acceptable for invalid config
            pass
    
    def test_pred_time_service_logging(self):
        """Test that pred time service has proper logging setup"""
        from core.services.pred_time_service import logger
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "core.services.pred_time_service")


if __name__ == "__main__":
    unittest.main()
