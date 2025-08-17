#!/usr/bin/env python3
"""
Tests for Pred_time module functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd


class TestPredTimeService:
    """Test PredTimeService functionality"""
    
    def test_service_initialization(self):
        """Test PredTimeService initialization"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        assert service.models_dir is not None
        assert hasattr(service, 'device')
        assert hasattr(service, 'news_service')
    
    def test_technical_indicators_calculation(self):
        """Test technical indicators calculation"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = {'technical_indicators': ['SMA', 'RSI', 'MACD', 'BB']}
        
        # Create sample data
        dates = pd.date_range('2025-01-01', periods=100, freq='5T')
        df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Calculate indicators
        result = service._calculate_technical_indicators(df)
        
        # Check that indicators were added
        assert 'SMA_5' in result.columns
        assert 'SMA_10' in result.columns
        assert 'SMA_20' in result.columns
        assert 'SMA_50' in result.columns
        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
        assert 'BB_middle' in result.columns
        assert 'price_change' in result.columns
        assert 'volume_ma' in result.columns
    
    def test_feature_preparation(self):
        """Test feature preparation for training"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = {
            'seq_len': 10,
            'pred_len': 5,
            'technical_indicators': ['SMA', 'RSI'],
            'feature_scaling': 'standard'
        }
        
        # Create sample data
        dates = pd.date_range('2025-01-01', periods=50, freq='5T')
        df = pd.DataFrame({
            'open': np.random.randn(50).cumsum() + 100,
            'high': np.random.randn(50).cumsum() + 102,
            'low': np.random.randn(50).cumsum() + 98,
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # Mock news service
        with patch.object(service, 'news_service'):
            X, y = service._prepare_features(df, 1)
            
            # Check output shapes
            assert len(X) > 0
            assert len(y) > 0
            assert X.shape[1] == service.config['seq_len']  # sequence length
            assert y.shape[1] == service.config['pred_len']  # prediction length
            assert X.shape[2] > 0  # feature size
    
    def test_model_creation_lstm(self):
        """Test LSTM model creation"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = {
            'model_type': 'LSTM',
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'pred_len': 5
        }
        service.feature_size = 10
        
        model = service._create_model()
        assert model is not None
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'fc')
    
    def test_model_creation_transformer(self):
        """Test Transformer model creation"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        service.config = {
            'model_type': 'Transformer',
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1,
            'pred_len': 5
        }
        service.feature_size = 10
        
        model = service._create_model()
        assert model is not None
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'output_projection')
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        from core.services.pred_time_service import PredTimeService
        
        service = PredTimeService()
        
        # Create sample predictions and targets
        predictions = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
        targets = np.array([0.15, -0.25, 0.35, -0.15, 0.25])
        
        metrics = service._calculate_metrics(predictions, targets)
        
        # Check that all metrics are calculated
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'direction_accuracy' in metrics
        assert 'correlation' in metrics
        assert 'r_squared' in metrics
        
        # Check metric types
        assert isinstance(metrics['rmse'], float)
        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['mape'], float)
        assert isinstance(metrics['direction_accuracy'], float)
        assert isinstance(metrics['correlation'], float)
        assert isinstance(metrics['r_squared'], float)
        
        # Check metric ranges
        assert 0 <= metrics['rmse']
        assert 0 <= metrics['mae']
        assert 0 <= metrics['mape']
        assert 0 <= metrics['direction_accuracy'] <= 1
        assert -1 <= metrics['correlation'] <= 1
        assert 0 <= metrics['r_squared'] <= 1


class TestPredTimeCeleryTasks:
    """Test Pred_time Celery tasks"""
    
    def test_train_pred_time_task_structure(self):
        """Test train_pred_time_task structure"""
        from backend.celery_app.tasks import train_pred_time_task
        
        # Verify task exists and is callable
        assert callable(train_pred_time_task)
        assert hasattr(train_pred_time_task, 'delay')
        assert hasattr(train_pred_time_task, 'apply_async')
    
    def test_evaluate_pred_time_task_structure(self):
        """Test evaluate_pred_time_task structure"""
        from backend.celery_app.tasks import evaluate_pred_time_task
        
        # Verify task exists and is callable
        assert callable(evaluate_pred_time_task)
        assert hasattr(evaluate_pred_time_task, 'delay')
        assert hasattr(evaluate_pred_time_task, 'apply_async')


class TestPredTimeAPIEndpoints:
    """Test Pred_time API endpoints"""
    
    @pytest.mark.asyncio
    async def test_train_pred_time_model_endpoint(self):
        """Test Pred_time model training endpoint"""
        from backend.app.routers.apidb_agent.router import train_pred_time_model
        
        # Mock dependencies
        mock_auth = "admin_token"
        
        # Mock request data
        request_data = {
            'coin_ids': [1, 2, 3],
            'agent_id': 1,
            'config': {
                'seq_len': 60,
                'pred_len': 12,
                'model_type': 'LSTM',
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100
            }
        }
        
        # Mock Celery
        with patch('backend.app.routers.apidb_agent.router.celery_app') as mock_celery:
            mock_task = Mock()
            mock_task.id = "test_task_id"
            mock_celery.send_task.return_value = mock_task
            
            result = await train_pred_time_model(request=request_data, _=mock_auth)
            
            assert result["status"] == "started"
            assert result["task_id"] == "test_task_id"
            assert len(result["config"]["coin_ids"]) == 3
            assert result["config"]["agent_id"] == 1
            assert result["config"]["seq_len"] == 60
            assert result["config"]["model_type"] == "LSTM"
    
    @pytest.mark.asyncio
    async def test_evaluate_pred_time_model_endpoint(self):
        """Test Pred_time model evaluation endpoint"""
        from backend.app.routers.apidb_agent.router import evaluate_pred_time_model
        
        # Mock dependencies
        mock_auth = "admin_token"
        
        # Mock request data
        request_data = {
            'model_path': '/path/to/model',
            'coin_ids': [1, 2],
            'evaluation_hours': 168
        }
        
        # Mock Celery
        with patch('backend.app.routers.apidb_agent.router.celery_app') as mock_celery:
            mock_task = Mock()
            mock_task.id = "test_task_id"
            mock_celery.send_task.return_value = mock_task
            
            result = await evaluate_pred_time_model(request=request_data, _=mock_auth)
            
            assert result["status"] == "started"
            assert result["task_id"] == "test_task_id"
            assert result["config"]["model_path"] == '/path/to/model'
            assert len(result["config"]["coin_ids"]) == 2
            assert result["config"]["evaluation_hours"] == 168
    
    @pytest.mark.asyncio
    async def test_get_pred_time_models_endpoint(self):
        """Test getting Pred_time models endpoint"""
        from backend.app.routers.apidb_agent.router import get_pred_time_models
        
        # Mock dependencies
        mock_auth = "admin_token"
        agent_id = 1
        
        # Mock file system operations
        with patch('pathlib.Path.exists') as mock_exists:
            with patch('pathlib.Path.iterdir') as mock_iterdir:
                mock_exists.return_value = True
                
                # Mock model directory
                mock_model_dir = Mock()
                mock_model_dir.is_dir.return_value = True
                mock_model_dir.name = "agent_1_20250622_120000"
                
                # Mock model files
                mock_model_file = Mock()
                mock_model_file.exists.return_value = True
                mock_model_file.stat.return_value = Mock(st_size=1024000)
                
                mock_config_file = Mock()
                mock_config_file.exists.return_value = True
                
                mock_metadata_file = Mock()
                mock_metadata_file.exists.return_value = True
                
                mock_model_dir.__truediv__.side_effect = lambda x: {
                    'model.pth': mock_model_file,
                    'config.json': mock_config_file,
                    'metadata.json': mock_metadata_file
                }[x]
                
                mock_iterdir.return_value = [mock_model_dir]
                
                # Mock file reading
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = '{"test": "data"}'
                    
                    result = await get_pred_time_models(agent_id=agent_id, _=mock_auth)
                    
                    assert result["status"] == "success"
                    assert result["agent_id"] == agent_id
                    assert "models" in result
                    assert result["count"] >= 0
    
    @pytest.mark.asyncio
    async def test_make_pred_time_prediction_endpoint(self):
        """Test making Pred_time prediction endpoint"""
        from backend.app.routers.apidb_agent.router import make_pred_time_prediction
        
        # Mock dependencies
        mock_auth = "admin_token"
        
        # Mock request data
        request_data = {
            'model_path': '/path/to/model',
            'coin_id': 1,
            'features': [[1.0, 2.0, 3.0, 4.0, 5.0] for _ in range(10)]
        }
        
        # Mock service and database
        with patch('backend.app.routers.apidb_agent.router.db_helper.get_session') as mock_db:
            with patch('core.services.pred_time_service.PredTimeService') as mock_service_class:
                mock_session = AsyncMock()
                mock_db.return_value.__aenter__.return_value = mock_session
                
                mock_service = Mock()
                mock_service_class.return_value = mock_service
                
                # Mock model loading
                mock_model = Mock()
                mock_service.load_model.return_value = mock_model
                
                # Mock prediction
                mock_prediction = {
                    'timestamp': datetime.now(),
                    'coin_id': 1,
                    'predicted_price': 100.5,
                    'predicted_change': 0.5,
                    'confidence': 0.8,
                    'direction': 'up',
                    'model_version': 'latest'
                }
                mock_service.predict.return_value = mock_prediction
                
                result = await make_pred_time_prediction(request=request_data, _=mock_auth)
                
                assert result["status"] == "success"
                assert "prediction" in result
                assert result["prediction"]["coin_id"] == 1
                assert result["prediction"]["direction"] == "up"


class TestPredTimeSchemas:
    """Test Pred_time Pydantic schemas"""
    
    def test_pred_time_train_config_validation(self):
        """Test PredTimeTrainConfig validation"""
        from backend.app.schemas.agent import PredTimeTrainConfig
        
        # Valid configuration
        valid_config = {
            'seq_len': 60,
            'pred_len': 12,
            'step_size': 1,
            'model_type': 'LSTM',
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 20,
            'technical_indicators': ['SMA', 'RSI', 'MACD', 'BB'],
            'news_integration': True,
            'feature_scaling': 'standard',
            'val_split': 0.2,
            'test_split': 0.1
        }
        
        config = PredTimeTrainConfig(**valid_config)
        assert config.seq_len == 60
        assert config.model_type == 'LSTM'
        assert config.hidden_size == 128
        assert 'SMA' in config.technical_indicators
    
    def test_pred_time_train_config_constraints(self):
        """Test PredTimeTrainConfig field constraints"""
        from backend.app.schemas.agent import PredTimeTrainConfig
        from pydantic import ValidationError
        
        # Test invalid seq_len
        with pytest.raises(ValidationError):
            PredTimeTrainConfig(
                seq_len=5,  # Too small
                pred_len=12,
                model_type='LSTM'
            )
        
        # Test invalid model_type
        with pytest.raises(ValidationError):
            PredTimeTrainConfig(
                seq_len=60,
                pred_len=12,
                model_type='InvalidModel'  # Invalid type
            )
        
        # Test invalid learning_rate
        with pytest.raises(ValidationError):
            PredTimeTrainConfig(
                seq_len=60,
                pred_len=12,
                model_type='LSTM',
                learning_rate=1.1  # Too large
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
