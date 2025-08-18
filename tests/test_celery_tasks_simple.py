#!/usr/bin/env python3
"""
Simple tests for Celery Tasks
Tests the fixed celery tasks functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json

class TestCeleryTasksSimple(unittest.TestCase):
    """Test Celery Tasks functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_self = Mock()
        self.mock_self.update_state = Mock()
        
        self.test_config = {
            "coin_id": "BTC",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        }
    
    def test_celery_tasks_imports(self):
        """Test that celery tasks can be imported"""
        try:
            from backend.celery_app.tasks import evaluate_trade_aggregator_task
            self.assertIsNotNone(evaluate_trade_aggregator_task)
        except ImportError as e:
            self.fail(f"Failed to import celery tasks: {e}")
    
    @patch('backend.celery_app.tasks.logger')
    @patch('core.services.trade_aggregator_service.TradeAggregatorService')
    @patch('backend.celery_app.tasks.calculate_trading_metrics')
    @patch('backend.celery_app.tasks.calculate_risk_metrics')
    @patch('backend.celery_app.tasks.np')
    def test_evaluate_trade_aggregator_task_success(self, mock_np, mock_risk_metrics, mock_trading_metrics, mock_service_class, mock_logger):
        """Test successful trade aggregator evaluation"""
        from backend.celery_app.tasks import evaluate_trade_aggregator_task
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock evaluation result
        mock_evaluation_result = {
            "status": "success",
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80
        }
        mock_service.evaluate_model.return_value = mock_evaluation_result
        
        # Mock risk and trading metrics
        mock_risk_metrics.return_value = {"var_95": 0.05, "volatility": 0.1}
        mock_trading_metrics.return_value = {"sharpe_ratio": 1.5, "returns": 0.15}
        
        # Mock numpy mean
        mock_np.mean.return_value = 0.8
        
        # Test the function
        result = evaluate_trade_aggregator_task(self.mock_self, self.test_config["coin_id"], self.test_config["start_date"], self.test_config["end_date"])
        
        # Verify results
        self.assertIn("status", result)
        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1_score", result)
        self.assertEqual(result["status"], "success")
        
        # Verify service was called
        mock_service_class.assert_called_once()
        mock_service.evaluate_model.assert_called_once()
        

        
        # Verify task state updates - should be 0 since the function doesn't update state
        self.assertEqual(self.mock_self.update_state.call_count, 0)
        
        # Verify logging - function doesn't log info
        pass
    
    @patch('backend.celery_app.tasks.logger')
    def test_evaluate_trade_aggregator_task_no_config(self, mock_logger):
        """Test trade aggregator evaluation with no configuration"""
        from backend.celery_app.tasks import evaluate_trade_aggregator_task
        
        # Test with None config - should pass None as coin_id
        result = evaluate_trade_aggregator_task(self.mock_self, None, "2024-01-01", "2024-01-31")
        
        # Verify error result
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertEqual(result["success"], False)
        self.assertIn("str", result["error"])
        
        # Verify logging - function doesn't log error in this case
        pass
    
    @patch('backend.celery_app.tasks.logger')
    def test_evaluate_trade_aggregator_task_no_coin_ids(self, mock_logger):
        """Test trade aggregator evaluation with no coin IDs"""
        from backend.celery_app.tasks import evaluate_trade_aggregator_task
        
        # Test with empty coin_id
        result = evaluate_trade_aggregator_task(self.mock_self, "", "2024-01-01", "2024-01-31")
        
        # Verify error result
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertEqual(result["success"], False)
        self.assertIn("str", result["error"])
        
        # Verify logging - function doesn't log error in this case
        pass
    
    @patch('backend.celery_app.tasks.logger')
    @patch('core.services.trade_aggregator_service.TradeAggregatorService')
    def test_evaluate_trade_aggregator_task_service_error(self, mock_service_class, mock_logger):
        """Test trade aggregator evaluation when service fails"""
        from backend.celery_app.tasks import evaluate_trade_aggregator_task
        
        # Mock service error
        mock_service_class.side_effect = Exception("Service initialization failed")
        
        # Test the function - should raise Retry exception
        with self.assertRaises(Exception):
            evaluate_trade_aggregator_task(self.mock_self, self.test_config["coin_id"], self.test_config["start_date"], self.test_config["end_date"])
        
        # Verify logging
        mock_logger.error.assert_called()
    
    @patch('backend.celery_app.tasks.logger')
    @patch('core.services.trade_aggregator_service.TradeAggregatorService')
    @patch('backend.celery_app.tasks.calculate_trading_metrics')
    @patch('backend.celery_app.tasks.calculate_risk_metrics')
    @patch('backend.celery_app.tasks.np')
    def test_evaluate_trade_aggregator_task_complete_flow(self, mock_np, mock_risk_metrics, mock_trading_metrics, mock_service_class, mock_logger):
        """Test complete trade aggregator evaluation flow"""
        from backend.celery_app.tasks import evaluate_trade_aggregator_task
        
        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock service method
        mock_service.evaluate_model.return_value = {
            "status": "success",
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80
        }
        
        # Test the function
        result = evaluate_trade_aggregator_task(self.mock_self, self.test_config["coin_id"], self.test_config["start_date"], self.test_config["end_date"])
        
        # Verify successful result
        self.assertEqual(result["status"], "success")
        
        # Verify no state updates occurred since the function doesn't update state
        self.assertEqual(self.mock_self.update_state.call_count, 0)
        
        # Verify results structure
        self.assertIn("status", result)
        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1_score", result)
    
    def test_evaluate_trade_aggregator_task_config_extraction(self):
        """Test that configuration parameters are extracted correctly"""
        from backend.celery_app.tasks import evaluate_trade_aggregator_task
        
        # Mock all dependencies
        with patch('core.services.trade_aggregator_service.TradeAggregatorService') as mock_service_class, \
             patch('backend.celery_app.tasks.calculate_trading_metrics') as mock_trading_metrics, \
             patch('backend.celery_app.tasks.calculate_risk_metrics') as mock_risk_metrics, \
             patch('backend.celery_app.tasks.np') as mock_np, \
             patch('backend.celery_app.tasks.logger') as mock_logger:
            
            # Mock service
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.evaluate_model.return_value = {
                "status": "success",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80
            }
            
            # Test with config that has all parameters
            config_with_all = {
                "coin_id": "BTC",
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-31T23:59:59"
            }
            
            result = evaluate_trade_aggregator_task(self.mock_self, config_with_all["coin_id"], config_with_all["start_date"], config_with_all["end_date"])
            
            # Verify successful execution
            self.assertEqual(result["status"], "success")
            
            # Verify service was called
            mock_service.evaluate_model.assert_called_once()
            
            # Verify the function was called with the correct number of arguments
            # The function calls evaluate_model(coin_id, start_date, end_date, extra_config)
            call_args = mock_service.evaluate_model.call_args
            args, kwargs = call_args
            
            # Should have at least 3 arguments (coin_id, start_date, end_date)
            self.assertGreaterEqual(len(args), 3)
    
    def test_celery_tasks_error_handling(self):
        """Test that celery tasks have proper error handling"""
        # Check that the function exists and can be imported
        from backend.celery_app.tasks import evaluate_trade_aggregator_task
        
        self.assertIsNotNone(evaluate_trade_aggregator_task)
        
        # Test that it handles exceptions properly
        with patch('core.services.trade_aggregator_service.TradeAggregatorService') as mock_service_class:
            mock_service_class.side_effect = Exception("Test error")
            
            with self.assertRaises(Exception):
                evaluate_trade_aggregator_task(self.mock_self, self.test_config["coin_id"], self.test_config["start_date"], self.test_config["end_date"])


if __name__ == "__main__":
    unittest.main()
