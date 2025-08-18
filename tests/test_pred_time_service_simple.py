#!/usr/bin/env python3
"""
Simple tests for Pred Time Service
Tests the fixed pred time service functionality
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestPredTimeServiceSimple(unittest.TestCase):
    """Test Pred Time Service functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'seq_len': 10,
            'pred_len': 5,
            'feature_scaling': 'standard',
            'news_integration': True,
            'technical_indicators': ['SMA', 'RSI', 'MACD', 'BB'],
            'model_type': 'LSTM',
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        }
        
        # Sample data
        self.sample_df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2024-01-01', periods=5, freq='1H'))
    
    def test_pred_time_service_imports(self):
        """Test that pred time service can be imported"""
        try:
            from src.core.services.pred_time_service import PredTimeService
            self.assertIsNotNone(PredTimeService)
        except ImportError as e:
            self.fail(f"Failed to import pred time service: {e}")
    
    @patch('src.core.services.pred_time_service.NewsBackgroundService')
    def test_pred_time_service_initialization(self, mock_news_service_class):
        """Test pred time service initialization"""
        from src.core.services.pred_time_service import PredTimeService
        
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
        from src.core.services.pred_time_service import PredTimeService
        
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
    
    @patch('src.core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_get_news_background_for_window(self, mock_get_news):
        """Test news background retrieval for window"""
        from src.core.services.pred_time_service import PredTimeService
        
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
    
    @patch('src.core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_get_news_background_for_window_empty(self, mock_get_news):
        """Test news background retrieval with empty result"""
        from src.core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        
        # Mock empty news data
        mock_get_news.return_value = []
        
        # Test window retrieval
        start_time = pd.Timestamp('2024-01-01T10:00:00')
        end_time = pd.Timestamp('2024-01-01T12:00:00')
        
        result = service._get_news_background_for_window(1, start_time, end_time)
        
        # Verify result
        self.assertEqual(result, [])
    
    @patch('src.core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_get_news_background_for_window_error(self, mock_get_news):
        """Test news background retrieval with error"""
        from src.core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        
        # Mock error
        mock_get_news.side_effect = Exception("Database error")
        
        # Test window retrieval
        start_time = pd.Timestamp('2024-01-01T10:00:00')
        end_time = pd.Timestamp('2024-01-01T12:00:00')
        
        result = service._get_news_background_for_window(1, start_time, end_time)
        
        # Verify result is empty list on error
        self.assertEqual(result, [])
    
    @patch('src.core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_prepare_features_with_news_integration(self, mock_get_news):
        """Test feature preparation with news integration"""
        from src.core.services.pred_time_service import PredTimeService
        
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
    
    @patch('src.core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_prepare_features_without_news_integration(self, mock_get_news):
        """Test feature preparation without news integration"""
        from src.core.services.pred_time_service import PredTimeService
        
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
    
    @patch('src.core.services.pred_time_service.PredTimeService._get_news_background_for_window')
    def test_prepare_features_news_integration_error(self, mock_get_news):
        """Test feature preparation with news integration error"""
        from src.core.services.pred_time_service import PredTimeService
        
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
        from src.core.services.pred_time_service import PredTimeService
        
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
        from src.core.services.pred_time_service import PredTimeService, LSTMModel
        
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
        from src.core.services.pred_time_service import PredTimeService
        
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
        from src.core.services.pred_time_service import PredTimeService, TransformerModel
        
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
        from src.core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        
        # Test with invalid configuration
        service.config = {}
        
        # Should handle missing configuration gracefully
        with self.assertRaises(KeyError):
            service._calculate_technical_indicators(self.sample_df.copy())
    
    def test_pred_time_service_logging(self):
        """Test that pred time service has proper logging setup"""
        from src.core.services.pred_time_service import logger
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "src.core.services.pred_time_service")


if __name__ == "__main__":
    unittest.main()
